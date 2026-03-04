# anypoint_publisher.py
# ─────────────────────────────────────────────────────────────────────────────
# Pushes RAML project files to Anypoint Platform Design Center.
# Includes pre-push validation so errors are caught before upload.
#
# Required headers on every Design Center request:
#   Authorization:      Bearer {token}
#   x-organization-id:  {org_id}
#   x-owner-id:         {user_id}   ← auto-extracted from login response
#
# .env keys:
#   ANYPOINT_USERNAME, ANYPOINT_PASSWORD, ANYPOINT_ORG_ID
#   ANYPOINT_OWNER_ID  (optional fallback if auto-extract fails)
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import json
import requests
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

ANYPOINT_BASE = "https://anypoint.mulesoft.com"
DESIGN_CENTER = f"{ANYPOINT_BASE}/designcenter/api-designer"

# Valid RAML 1.0 security scheme types
VALID_SECURITY_TYPES = {
    "OAuth 2.0", "Basic Authentication",
    "Digest Authentication", "Pass Through",
}


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AnypointConfig:
    username: str
    password: str
    org_id:   str

    @classmethod
    def from_env(cls) -> "AnypointConfig":
        username = os.getenv("ANYPOINT_USERNAME")
        password = os.getenv("ANYPOINT_PASSWORD")
        org_id   = os.getenv("ANYPOINT_ORG_ID")
        missing  = [k for k, v in {
            "ANYPOINT_USERNAME": username,
            "ANYPOINT_PASSWORD": password,
            "ANYPOINT_ORG_ID":   org_id,
        }.items() if not v]
        if missing:
            raise ValueError(f"Missing .env keys: {', '.join(missing)}")
        return cls(username=username, password=password, org_id=org_id)


# ── Pre-push RAML validator ────────────────────────────────────────────────────

class RAMLValidator:
    """
    Lightweight pre-push validator for common RAML 1.0 errors.
    Catches the most frequent Design Center rejections before upload.
    Returns a list of {file, line, severity, message} dicts.
    """

    def validate(self, files: dict[str, str]) -> list[dict]:
        errors = []
        raml_files = {p: c for p, c in files.items() if p.endswith(".raml")}
        root = self._find_root(raml_files)

        for path, content in raml_files.items():
            errors.extend(self._check_file(path, content, root, files))

        # Cross-file checks
        errors.extend(self._check_includes(root, files) if root else [])
        errors.extend(self._check_exchange_json(files))

        return errors

    def _find_root(self, raml_files: dict) -> str | None:
        """Root file is the one that starts with #%RAML 1.0 (no fragment suffix)."""
        for path, content in raml_files.items():
            first = content.strip().split("\n")[0].strip()
            if first == "#%RAML 1.0":
                return path
        return None

    def _check_file(self, path: str, content: str, root: str, all_files: dict) -> list[dict]:
        errors = []
        lines  = content.split("\n")

        for i, line in enumerate(lines, 1):
            ln = line.strip()

            # 1. Wrong security scheme type
            if re.match(r"^\s*type:\s*(API Key|apiKey|api_key|ApiKey)\s*$", line, re.I):
                errors.append(self._err(path, i, "error",
                    f"Invalid security scheme type '{ln.split(':',1)[1].strip()}'. "
                    f"Valid types: OAuth 2.0, Basic Authentication, Digest Authentication, Pass Through, x-custom"))

            # 2. Traits defined inline in root instead of separate file
            if path == root and re.match(r"^traits:\s*$", line):
                errors.append(self._err(path, i, "error",
                    "Traits must be in a separate .raml file and imported via 'uses:', "
                    "not defined inline in the root file"))

            # 3. Fragment used with !include where uses: is needed
            if re.match(r"^\s*\w+:\s*!include\s+traits/", line):
                errors.append(self._err(path, i, "error",
                    "Trait files must be imported with 'uses:' not '!include'. "
                    "Example: uses:\\n  MyTraits: traits/common-traits.raml"))

            # 4. Trait file missing correct header
            if path != root and "traits/" in path:
                first = lines[0].strip() if lines else ""
                if not first.startswith("#%RAML 1.0 Trait"):
                    errors.append(self._err(path, 1, "error",
                        f"Trait file must start with '#%RAML 1.0 Trait', found: '{first}'"))
                break  # only check header once per file

            # 5. DataType file missing correct header
            if path != root and ("data-types/" in path or "types/" in path):
                first = lines[0].strip() if lines else ""
                if not first.startswith("#%RAML 1.0 DataType"):
                    errors.append(self._err(path, 1, "error",
                        f"DataType file must start with '#%RAML 1.0 DataType', found: '{first}'"))
                break

            # 6. !include referencing a file that doesn't exist
            inc_match = re.search(r"!include\s+(\S+)", line)
            if inc_match:
                inc_path = inc_match.group(1)
                # Resolve relative to the current file's directory
                base_dir  = "/".join(path.split("/")[:-1])
                resolved  = f"{base_dir}/{inc_path}".lstrip("/") if base_dir else inc_path
                # Normalise ../ segments
                parts = []
                for part in resolved.split("/"):
                    if part == "..":
                        if parts: parts.pop()
                    elif part and part != ".":
                        parts.append(part)
                resolved = "/".join(parts)
                if resolved not in all_files:
                    errors.append(self._err(path, i, "error",
                        f"!include target '{inc_path}' not found in project (resolved: '{resolved}')"))

        return errors

    def _check_includes(self, root: str, files: dict) -> list[dict]:
        """Warn about files that exist but are never referenced by root."""
        errors   = []
        content  = files.get(root, "")
        for path in files:
            if path in (root, "README.md", "exchange.json"):
                continue
            name = path.split("/")[-1]
            # Check if referenced anywhere in root (by path fragment or name)
            if path not in content and name not in content:
                errors.append(self._err(path, 1, "warning",
                    f"File is not referenced from the root file '{root}'. "
                    "Add it via !include or uses: or it will be ignored."))
        return errors

    def _check_exchange_json(self, files: dict) -> list[dict]:
        errors = []
        if "exchange.json" in files:
            try:
                ex = json.loads(files["exchange.json"])
                for field in ["groupId", "assetId", "version", "classifier"]:
                    if not ex.get(field):
                        errors.append(self._err("exchange.json", 1, "error",
                            f"Missing required field: '{field}'"))
            except json.JSONDecodeError as e:
                errors.append(self._err("exchange.json", 1, "error",
                    f"Invalid JSON: {e}"))
        return errors

    @staticmethod
    def _err(file: str, line: int, severity: str, message: str) -> dict:
        return {"file": file, "line": line, "severity": severity, "message": message}


# ── Publisher ─────────────────────────────────────────────────────────────────

class AnypointPublisher:
    """
    Validates then pushes {path: content} files to Anypoint Design Center.

    Usage:
        publisher = AnypointPublisher(AnypointConfig.from_env())
        result    = publisher.push("Orders API", session.files)
    """

    def __init__(self, config: AnypointConfig, verbose: bool = False):
        self.config    = config
        self.verbose   = verbose
        self._token    = None
        self._user_id  = None
        self.validator = RAMLValidator()

    # ── Auth ──────────────────────────────────────────────────────────────────

    def _login(self):
        """POST /accounts/login — extracts token + user_id."""
        resp = requests.post(
            f"{ANYPOINT_BASE}/accounts/login",
            json={"username": self.config.username, "password": self.config.password},
            timeout=15,
        )
        if not resp.ok:
            raise ValueError(f"Login failed ({resp.status_code}): {resp.text[:300]}")

        body    = resp.json()
        token   = body.get("access_token")
        user_id = (body.get("user", {}).get("id")
                   or body.get("userId")
                   or body.get("user_id")
                   or os.getenv("ANYPOINT_OWNER_ID"))

        if not token:
            raise ValueError(f"No access_token in login response")
        if not user_id:
            raise ValueError(
                f"Could not extract user_id (keys: {list(body.keys())}). "
                "Add ANYPOINT_OWNER_ID=<uuid> to .env. "
                "Find it: Anypoint > Access Management > Users > click your user > copy UUID from URL."
            )
        self._token   = token
        self._user_id = str(user_id)
        if self.verbose:
            print(f"[Publisher] Authenticated ✓  user={self._user_id[:8]}…")

    def _headers(self) -> dict:
        if not self._token:
            self._login()
        return {
            "Authorization":     f"Bearer {self._token}",
            "Content-Type":      "application/json",
            "x-organization-id": self.config.org_id,
            "x-owner-id":        self._user_id,
        }

    # ── Design Center steps ────────────────────────────────────────────────────

    def _create_project(self, name: str) -> str:
        resp = requests.post(f"{DESIGN_CENTER}/projects",
            headers=self._headers(), json={"name": name, "classifier": "raml"}, timeout=15)
        if resp.status_code == 409:
            raise ValueError(
                f"Project '{name}' already exists. Use a different name or add a version suffix.")
        if not resp.ok:
            raise ValueError(f"Create project failed ({resp.status_code}): {resp.text[:300]}")
        pid = resp.json().get("id")
        if not pid:
            raise ValueError(f"No project id in response: {resp.text[:200]}")
        if self.verbose: print(f"[Publisher] Project created: {pid}")
        return pid

    def _acquire_lock(self, pid: str):
        resp = requests.post(
            f"{DESIGN_CENTER}/projects/{pid}/branches/master/acquireLock",
            headers=self._headers(), json={}, timeout=15)
        if not resp.ok:
            raise ValueError(f"acquireLock failed ({resp.status_code}): {resp.text[:200]}")
        if self.verbose: print("[Publisher] Lock acquired ✓")

    def _upload_files(self, pid: str, files: dict[str, str]) -> list:
        payload = [
            {
                "path":    path,
                "type":    "FILE",
                "content": content,
                "title":   path.split("/")[-1].replace(".raml","").replace("-"," ").title(),
            }
            for path, content in files.items()
        ]
        resp = requests.post(
            f"{DESIGN_CENTER}/projects/{pid}/branches/master/save",
            headers=self._headers(), json=payload, timeout=30)
        if not resp.ok:
            raise ValueError(f"Upload failed ({resp.status_code}): {resp.text[:300]}")
        if self.verbose: print(f"[Publisher] {len(payload)} files uploaded ✓")
        return resp.json()

    def _release_lock(self, pid: str):
        try:
            requests.post(
                f"{DESIGN_CENTER}/projects/{pid}/branches/master/releaseLock",
                headers=self._headers(), json={}, timeout=10)
            if self.verbose: print("[Publisher] Lock released ✓")
        except Exception as e:
            if self.verbose: print(f"[Publisher] releaseLock skipped: {e}")

    # ── exchange.json auto-generation ──────────────────────────────────────────

    @staticmethod
    def _make_exchange_json(project_name: str, org_id: str) -> str:
        """
        Generate a valid exchange.json required by Anypoint.
        Without this file (or with a malformed one) Design Center shows
        'should have required property groupId' errors.
        """
        asset_id = re.sub(r"[^a-z0-9-]", "-", project_name.lower()).strip("-")
        return json.dumps({
            "groupId":    org_id,
            "assetId":    asset_id,
            "version":    "1.0.0",
            "classifier": "raml",
            "name":       project_name,
        }, indent=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def validate(self, files: dict[str, str]) -> list[dict]:
        """
        Run pre-push validation. Returns list of {file, line, severity, message}.
        Call this before push() to surface errors in the UI.
        """
        return self.validator.validate(files)

    def push(self, project_name: str, files: dict[str, str],
             skip_validation: bool = False) -> dict:
        """
        Validate then push all files to Anypoint Design Center.

        Returns:
            {
                project_id, project_name, project_url,
                files_pushed, file_count,
                validation_errors: []   ← empty if all clear
            }

        Raises ValueError if validation finds errors (unless skip_validation=True).
        """
        if not files:
            raise ValueError("No files to push.")

        # Always inject a valid exchange.json — never use agent-generated one
        files = dict(files)  # don't mutate caller's dict
        files.pop("exchange.json", None)
        files["exchange.json"] = self._make_exchange_json(project_name, self.config.org_id)

        # Pre-push validation
        validation_errors = self.validator.validate(files)
        hard_errors = [e for e in validation_errors if e["severity"] == "error"]
        if hard_errors and not skip_validation:
            raise ValueError(
                f"Validation failed with {len(hard_errors)} error(s). "
                "Fix them or use skip_validation=True to push anyway."
            )

        # Authenticate
        self._login()

        # Push
        pid      = self._create_project(project_name)
        self._acquire_lock(pid)
        uploaded = self._upload_files(pid, files)
        self._release_lock(pid)

        url = f"https://anypoint.mulesoft.com/designcenter/api-designer/projects/{pid}"
        return {
            "project_id":        pid,
            "project_name":      project_name,
            "project_url":       url,
            "files_pushed":      [f["path"] for f in uploaded if f.get("type") == "FILE"],
            "file_count":        len(files),
            "validation_errors": validation_errors,
        }

    def list_projects(self) -> list[dict]:
        resp = requests.get(f"{DESIGN_CENTER}/projects",
            headers=self._headers(), timeout=15)
        resp.raise_for_status()
        return resp.json()

    def delete_project(self, project_id: str):
        resp = requests.delete(f"{DESIGN_CENTER}/projects/{project_id}",
            headers=self._headers(), timeout=15)
        resp.raise_for_status()
# anypoint_publisher.py
# ─────────────────────────────────────────────────────────────────────────────
# Pushes a generated RAML project to Anypoint Platform Design Center.
#
# Required headers on every Design Center request (source: MuleSoft docs):
#   Authorization:      Bearer {token}
#   x-organization-id:  {org_id}        — from .env / Access Management
#   x-owner-id:         {user_id}        — extracted automatically from login
#
# The 500 error from the previous version was caused by the missing x-owner-id.
# The user_id is returned directly in the /accounts/login response, so no
# extra API call or manual lookup is needed.
#
# Flow:
#   1. POST /accounts/login              → token + user_id (auto-extracted)
#   2. POST /designcenter/.../projects   → project_id
#   3. POST .../branches/master/acquireLock
#   4. POST .../branches/master/save     → upload all files
#   5. POST .../branches/master/releaseLock
#
# .env keys required:
#   ANYPOINT_USERNAME, ANYPOINT_PASSWORD, ANYPOINT_ORG_ID
# ─────────────────────────────────────────────────────────────────────────────

import os
import requests
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

ANYPOINT_BASE = "https://anypoint.mulesoft.com"
DESIGN_CENTER = f"{ANYPOINT_BASE}/designcenter/api-designer"


@dataclass
class AnypointConfig:
    """
    Credentials for Anypoint Platform.

    Required .env keys:
        ANYPOINT_USERNAME   — Anypoint login username or email
        ANYPOINT_PASSWORD   — Anypoint login password
        ANYPOINT_ORG_ID     — Organization UUID (Access Management → Organization page)

    x-owner-id (user UUID) is extracted automatically from the login response.
    You do NOT need to add it to .env.
    """
    username: str
    password: str
    org_id:   str

    @classmethod
    def from_env(cls) -> "AnypointConfig":
        username = os.getenv("ANYPOINT_USERNAME")
        password = os.getenv("ANYPOINT_PASSWORD")
        org_id   = os.getenv("ANYPOINT_ORG_ID")
        missing  = [k for k, v in {"ANYPOINT_USERNAME": username,
                                    "ANYPOINT_PASSWORD": password,
                                    "ANYPOINT_ORG_ID":   org_id}.items() if not v]
        if missing:
            raise ValueError(f"Missing .env keys: {', '.join(missing)}")
        return cls(username=username, password=password, org_id=org_id)


class AnypointPublisher:
    """
    Publishes {path: content} files to Anypoint Platform Design Center.

    Usage:
        publisher = AnypointPublisher(AnypointConfig.from_env())
        result    = publisher.publish("Orders API", session.files)
        print(result["project_url"])
    """

    def __init__(self, config: AnypointConfig, verbose: bool = False):
        self.config   = config
        self.verbose  = verbose
        self._token   = None
        self._user_id = None     # extracted from login, used as x-owner-id

    # ── Step 1: Authenticate ──────────────────────────────────────────────────

    def _login(self):
        """
        POST /accounts/login
        Extracts both access_token and user_id from the response.
        user_id is required as x-owner-id on every Design Center request.
        """
        resp = requests.post(
            f"{ANYPOINT_BASE}/accounts/login",
            json={"username": self.config.username, "password": self.config.password},
            timeout=15,
        )
        # Surface useful error text instead of a bare HTTP error
        if not resp.ok:
            raise ValueError(
                f"Anypoint login failed ({resp.status_code}): {resp.text[:300]}"
            )

        body = resp.json()
        token   = body.get("access_token")
        user_id = body.get("user", {}).get("id") or body.get("userId") or body.get("user_id")

        if not token:
            raise ValueError(f"No access_token in login response: {resp.text[:300]}")
        if not user_id:
            # Last resort: allow manual override in .env
            user_id = os.getenv("ANYPOINT_OWNER_ID")
        if not user_id:
            raise ValueError(
                f"Could not extract user_id from login response "
                f"(keys: {list(body.keys())}). "
                f"Add ANYPOINT_OWNER_ID=<your-user-uuid> to .env. "
                f"Find it at: Anypoint > Access Management > Users > click your user > copy UUID from URL."
            )

        self._token   = token
        self._user_id = str(user_id)

        if self.verbose:
            print(f"[AnypointPublisher] Authenticated ✓  user_id={self._user_id}")

    def _headers(self) -> dict:
        """Build headers required by every Design Center API call."""
        if not self._token:
            self._login()
        return {
            "Authorization":     f"Bearer {self._token}",
            "Content-Type":      "application/json",
            "x-organization-id": self.config.org_id,
            "x-owner-id":        self._user_id,   # ← was missing, caused 500
        }

    # ── Step 2: Create project ────────────────────────────────────────────────

    def _create_project(self, project_name: str) -> str:
        """POST /projects — returns the new project_id."""
        resp = requests.post(
            f"{DESIGN_CENTER}/projects",
            headers=self._headers(),
            json={"name": project_name, "classifier": "raml"},
            timeout=15,
        )
        if resp.status_code == 409:
            raise ValueError(
                f"A project named '{project_name}' already exists in Design Center. "
                "Use a different name or add a version suffix (e.g. 'Orders API v2')."
            )
        if not resp.ok:
            raise ValueError(
                f"Create project failed ({resp.status_code}): {resp.text[:300]}"
            )
        project_id = resp.json().get("id")
        if not project_id:
            raise ValueError(f"No project id in response: {resp.text[:300]}")
        if self.verbose:
            print(f"[AnypointPublisher] Project created: {project_id}")
        return project_id

    # ── Step 3: Acquire write lock ────────────────────────────────────────────

    def _acquire_lock(self, project_id: str):
        """POST .../branches/master/acquireLock — required before saving files."""
        resp = requests.post(
            f"{DESIGN_CENTER}/projects/{project_id}/branches/master/acquireLock",
            headers=self._headers(),
            json={},
            timeout=15,
        )
        if not resp.ok:
            raise ValueError(
                f"acquireLock failed ({resp.status_code}): {resp.text[:300]}"
            )
        if self.verbose:
            print("[AnypointPublisher] Write lock acquired ✓")

    # ── Step 4: Upload files ──────────────────────────────────────────────────

    def _upload_files(self, project_id: str, files: dict[str, str]) -> list:
        """
        POST .../branches/master/save
        Sends all project files in a single request.
        Each entry: {path, type, content, title}
        """
        payload = [
            {
                "path":    path,
                "type":    "FILE",
                "content": content,
                "title":   path.split("/")[-1].replace(".raml", "").replace("-", " ").title(),
            }
            for path, content in files.items()
        ]
        resp = requests.post(
            f"{DESIGN_CENTER}/projects/{project_id}/branches/master/save",
            headers=self._headers(),
            json=payload,
            timeout=30,
        )
        if not resp.ok:
            raise ValueError(
                f"File upload failed ({resp.status_code}): {resp.text[:300]}"
            )
        if self.verbose:
            print(f"[AnypointPublisher] {len(payload)} files uploaded ✓")
        return resp.json()

    # ── Step 5: Release lock ──────────────────────────────────────────────────

    def release_lock(self, project_id: str):
        """Release write lock so collaborators can edit in Design Center."""
        try:
            resp = requests.post(
                f"{DESIGN_CENTER}/projects/{project_id}/branches/master/releaseLock",
                headers=self._headers(),
                json={},
                timeout=10,
            )
            if self.verbose and resp.ok:
                print("[AnypointPublisher] Write lock released ✓")
        except Exception as e:
            if self.verbose:
                print(f"[AnypointPublisher] releaseLock skipped: {e}")

    # ── Public: publish ───────────────────────────────────────────────────────

    def publish(self, project_name: str, files: dict[str, str]) -> dict:
        """
        Full publish flow: login → create project → lock → upload → release lock.

        Returns:
            {
                "project_id":   "...",
                "project_name": "...",
                "project_url":  "https://anypoint.mulesoft.com/designcenter/...",
                "files_pushed": ["api.raml", ...],
                "file_count":   N,
            }
        """
        if not files:
            raise ValueError("No files to publish.")

        # Login first so _user_id is populated before any other call
        self._login()

        project_id = self._create_project(project_name)
        self._acquire_lock(project_id)
        uploaded   = self._upload_files(project_id, files)
        self.release_lock(project_id)

        project_url = (
            f"https://anypoint.mulesoft.com/designcenter/api-designer/projects/{project_id}"
        )
        return {
            "project_id":   project_id,
            "project_name": project_name,
            "project_url":  project_url,
            "files_pushed": [f["path"] for f in uploaded if f.get("type") == "FILE"],
            "file_count":   len(files),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def list_projects(self) -> list[dict]:
        """GET /projects — list all Design Center projects for this org."""
        resp = requests.get(
            f"{DESIGN_CENTER}/projects",
            headers=self._headers(),
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()

    def delete_project(self, project_id: str):
        """DELETE /projects/{id} — permanently removes the project."""
        resp = requests.delete(
            f"{DESIGN_CENTER}/projects/{project_id}",
            headers=self._headers(),
            timeout=15,
        )
        resp.raise_for_status()
        if self.verbose:
            print(f"[AnypointPublisher] Project {project_id} deleted ✓")
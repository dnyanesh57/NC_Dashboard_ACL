from __future__ import annotations

import os
import secrets
import smtplib
from datetime import datetime, timedelta
from email.message import EmailMessage
from typing import List, Optional, Tuple, Dict, Set

from sqlalchemy import (
    create_engine, Column, Integer, String, Boolean, DateTime, ForeignKey, UniqueConstraint, select, func
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session
from sqlalchemy import inspect

try:
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - fallback when not running in Streamlit
    st = None  # type: ignore

try:
    from passlib.hash import bcrypt
except Exception:  # pragma: no cover
    bcrypt = None  # type: ignore


Base = declarative_base()


class User(Base):
    __tablename__ = "acl_users"
    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), nullable=False, default="user")  # 'admin' or 'user'
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    groups = relationship("UserGroup", back_populates="user", cascade="all, delete-orphan")
    user_viz = relationship("UserVisualization", back_populates="user", cascade="all, delete-orphan")


class Group(Base):
    __tablename__ = "acl_groups"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)

    members = relationship("UserGroup", back_populates="group", cascade="all, delete-orphan")
    group_viz = relationship("GroupVisualization", back_populates="group", cascade="all, delete-orphan")


class UserGroup(Base):
    __tablename__ = "acl_user_groups"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("acl_users.id", ondelete="CASCADE"), nullable=False)
    group_id = Column(Integer, ForeignKey("acl_groups.id", ondelete="CASCADE"), nullable=False)
    __table_args__ = (UniqueConstraint('user_id', 'group_id', name='uq_user_group'),)

    user = relationship("User", back_populates="groups")
    group = relationship("Group", back_populates="members")


class Visualization(Base):
    __tablename__ = "acl_visualizations"
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False)  # e.g., 'overview', 'status', ...
    name = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    user_binds = relationship("UserVisualization", back_populates="viz", cascade="all, delete-orphan")
    group_binds = relationship("GroupVisualization", back_populates="viz", cascade="all, delete-orphan")


class UserVisualization(Base):
    __tablename__ = "acl_user_visualizations"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("acl_users.id", ondelete="CASCADE"), nullable=False)
    viz_id = Column(Integer, ForeignKey("acl_visualizations.id", ondelete="CASCADE"), nullable=False)
    allowed = Column(Boolean, default=True, nullable=False)
    __table_args__ = (UniqueConstraint('user_id', 'viz_id', name='uq_user_viz'),)

    user = relationship("User", back_populates="user_viz")
    viz = relationship("Visualization", back_populates="user_binds")


class GroupVisualization(Base):
    __tablename__ = "acl_group_visualizations"
    id = Column(Integer, primary_key=True)
    group_id = Column(Integer, ForeignKey("acl_groups.id", ondelete="CASCADE"), nullable=False)
    viz_id = Column(Integer, ForeignKey("acl_visualizations.id", ondelete="CASCADE"), nullable=False)
    allowed = Column(Boolean, default=True, nullable=False)
    __table_args__ = (UniqueConstraint('group_id', 'viz_id', name='uq_group_viz'),)

    group = relationship("Group", back_populates="group_viz")
    viz = relationship("Visualization", back_populates="group_binds")


class PasswordResetToken(Base):
    __tablename__ = "acl_password_reset_tokens"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("acl_users.id", ondelete="CASCADE"), nullable=False)
    token = Column(String(255), unique=True, nullable=False, index=True)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Site(Base):
    __tablename__ = "acl_sites"
    id = Column(Integer, primary_key=True)
    key = Column(String(255), unique=True, nullable=False, index=True)  # expected to match 'Project Name'
    name = Column(String(255), nullable=False)

    members = relationship("UserSite", back_populates="site", cascade="all, delete-orphan")


class UserSite(Base):
    __tablename__ = "acl_user_sites"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("acl_users.id", ondelete="CASCADE"), nullable=False)
    site_id = Column(Integer, ForeignKey("acl_sites.id", ondelete="CASCADE"), nullable=False)
    __table_args__ = (UniqueConstraint('user_id', 'site_id', name='uq_user_site'),)

    user = relationship("User")
    site = relationship("Site", back_populates="members")


DEFAULT_VISUALIZATIONS: List[Tuple[str, str]] = [
    ("overview", "Overview"),
    ("status", "Status"),
    ("project_status", "Project Status"),
    ("project_explorer", "Project Explorer"),
    ("tower_wise", "Tower-Wise"),
    ("user_wise", "User-Wise"),
    ("activity_wise", "Activity-Wise"),
    ("timelines", "Timelines"),
    ("nc_view", "NC-View"),
    ("sketch_view", "Sketch-View"),
    ("nc_table", "NC Table"),
]


_engine = None
_SessionLocal = None


def _get_db_url_from_secrets() -> Optional[str]:
    # Preferred keys in Streamlit secrets
    keys = ["DB_URL", "db_url", "DATABASE_URL"]
    if st is not None:
        try:
            for k in keys:
                if k in st.secrets:
                    return str(st.secrets[k])
        except Exception:
            pass
    # Fallback to env vars
    for k in ["DB_URL", "DATABASE_URL"]:
        if k in os.environ:
            return os.environ[k]
    return None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        db_url = _get_db_url_from_secrets()
        if not db_url:
            raise RuntimeError("DB_URL not found in Streamlit secrets or environment.")
        # Ensure driver prefix for SQLAlchemy if user provided 'postgresql://'
        if db_url.startswith("postgresql://") and "+" not in db_url.split(":")[0]:
            db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        if "sslmode" not in db_url:
            sep = "&" if "?" in db_url else "?"
            db_url = f"{db_url}{sep}sslmode=require"
        _engine = create_engine(db_url, pool_pre_ping=True)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def get_session() -> Session:
    _ = get_engine()
    return _SessionLocal()


def init_db(seed_admin: bool = True) -> None:
    engine = get_engine()

    # Attempt to repair any partially-created/misconfigured tables from prior runs
    try:
        _repair_incorrect_tables(engine)
    except Exception:
        pass

    Base.metadata.create_all(engine)
    # Seed visualizations
    with get_session() as s:
        existing = {k for (k,) in s.execute(select(Visualization.key)).all()}
        for key, name in DEFAULT_VISUALIZATIONS:
            if key not in existing:
                s.add(Visualization(key=key, name=name, is_active=True))
        s.commit()

    if seed_admin:
        # Optional: seed first admin from secrets
        admin_email = None
        admin_password = None
        admin_name = None
        if st is not None:
            try:
                admin_email = st.secrets.get("ADMIN_EMAIL")
                admin_password = st.secrets.get("ADMIN_PASSWORD")
                admin_name = st.secrets.get("ADMIN_NAME", "Admin")
            except Exception:
                pass
        if admin_email and admin_password:
            ensure_admin(admin_email, admin_password, admin_name or "Admin")


def _repair_incorrect_tables(engine) -> None:
    """Drop known broken artifacts to allow a clean create_all.
    Specifically handles the case where 'acl_users' exists without an 'id' column.
    Safe to run repeatedly.
    """
    insp = inspect(engine)
    existing = set(insp.get_table_names())
    if "acl_users" in existing:
        try:
            cols = [c.get("name") for c in insp.get_columns("acl_users")]
        except Exception:
            cols = []
        if "id" not in cols:
            # Drop the malformed table and any dependents
            with engine.begin() as conn:
                conn.exec_driver_sql("DROP TABLE IF EXISTS acl_users CASCADE")


def list_sites() -> List[Dict]:
    with get_session() as s:
        rows = s.execute(select(Site).order_by(Site.name.asc())).scalars().all()
        return [{"id": r.id, "key": r.key, "name": r.name} for r in rows]


def create_site(key: str, name: Optional[str] = None) -> Tuple[bool, str]:
    key = (key or "").strip()
    if not key:
        return False, "Invalid site key"
    with get_session() as s:
        ex = s.execute(select(Site).where(func.lower(Site.key) == key.lower())).scalar_one_or_none()
        if ex:
            return False, "Site already exists"
        site = Site(key=key, name=(name or key).strip())
        s.add(site)
        s.commit()
        return True, "Site created"


def delete_site(key: str) -> Tuple[bool, str]:
    with get_session() as s:
        site = s.execute(select(Site).where(func.lower(Site.key) == key.lower().strip())).scalar_one_or_none()
        if not site:
            return False, "Site not found"
        s.delete(site)
        s.commit()
        return True, "Site deleted"


def assign_user_to_site(user_email: str, site_key: str) -> Tuple[bool, str]:
    with get_session() as s:
        u = s.execute(select(User).where(User.email == user_email.lower().strip())).scalar_one_or_none()
        site = s.execute(select(Site).where(func.lower(Site.key) == site_key.lower().strip())).scalar_one_or_none()
        if not u or not site:
            return False, "User or Site not found"
        ex = s.execute(select(UserSite).where(UserSite.user_id == u.id, UserSite.site_id == site.id)).scalar_one_or_none()
        if ex:
            return True, "Already assigned"
        s.add(UserSite(user_id=u.id, site_id=site.id))
        s.commit()
        return True, "Assigned"


def unassign_user_from_site(user_email: str, site_key: str) -> Tuple[bool, str]:
    with get_session() as s:
        u = s.execute(select(User).where(User.email == user_email.lower().strip())).scalar_one_or_none()
        site = s.execute(select(Site).where(func.lower(Site.key) == site_key.lower().strip())).scalar_one_or_none()
        if not u or not site:
            return False, "User or Site not found"
        link = s.execute(select(UserSite).where(UserSite.user_id == u.id, UserSite.site_id == site.id)).scalar_one_or_none()
        if not link:
            return False, "Not assigned"
        s.delete(link)
        s.commit()
        return True, "Unassigned"


def list_user_sites(user_email: str) -> List[str]:
    with get_session() as s:
        u = s.execute(select(User).where(User.email == user_email.lower().strip())).scalar_one_or_none()
        if not u:
            return []
        rows = s.execute(
            select(Site.key)
            .join(UserSite, UserSite.site_id == Site.id)
            .where(UserSite.user_id == u.id)
        ).all()
        return [k for (k,) in rows]


def allowed_sites_for(user_email: str) -> Set[str]:
    with get_session() as s:
        u = s.execute(select(User).where(User.email == user_email.lower().strip())).scalar_one_or_none()
        if not u:
            return set()
        if u.role == "admin":
            return set()
        rows = s.execute(
            select(Site.key)
            .join(UserSite, UserSite.site_id == Site.id)
            .where(UserSite.user_id == u.id)
        ).all()
        return {k for (k,) in rows}


def _hash_password(password: str) -> str:
    if not bcrypt:
        raise RuntimeError("passlib[bcrypt] is required. Please add to requirements.txt")
    return bcrypt.hash(password)


def _verify_password(password: str, password_hash: str) -> bool:
    if not bcrypt:
        raise RuntimeError("passlib[bcrypt] is required. Please add to requirements.txt")
    try:
        return bcrypt.verify(password, password_hash)
    except Exception:
        return False


def ensure_admin(email: str, password: str, name: str = "Admin") -> None:
    with get_session() as s:
        user = s.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if user is None:
            user = User(email=email.lower().strip(), name=name, password_hash=_hash_password(password), role="admin", is_active=True)
            s.add(user)
            s.commit()


def create_user(email: str, name: str, password: str, role: str = "user") -> Tuple[bool, str]:
    email = email.lower().strip()
    if not email or "@" not in email:
        return False, "Invalid email"
    if role not in ("user", "admin"):
        return False, "Invalid role"
    with get_session() as s:
        existing = s.execute(select(User).where(User.email == email)).scalar_one_or_none()
        if existing:
            return False, "User already exists"
        u = User(email=email, name=name.strip() or email.split("@")[0], password_hash=_hash_password(password), role=role, is_active=True)
        s.add(u)
        s.commit()
        return True, "User created"


def delete_user(email: str) -> Tuple[bool, str]:
    with get_session() as s:
        user = s.execute(select(User).where(User.email == email.lower().strip())).scalar_one_or_none()
        if not user:
            return False, "User not found"
        s.delete(user)
        s.commit()
        return True, "User deleted"


def set_password(email: str, new_password: str) -> Tuple[bool, str]:
    with get_session() as s:
        user = s.execute(select(User).where(User.email == email.lower().strip())).scalar_one_or_none()
        if not user:
            return False, "User not found"
        user.password_hash = _hash_password(new_password)
        s.commit()
        return True, "Password updated"


def set_active(email: str, active: bool) -> Tuple[bool, str]:
    with get_session() as s:
        user = s.execute(select(User).where(User.email == email.lower().strip())).scalar_one_or_none()
        if not user:
            return False, "User not found"
        user.is_active = active
        s.commit()
        return True, "Updated"


def authenticate(email: str, password: str) -> Optional[Dict]:
    with get_session() as s:
        u = s.execute(select(User).where(User.email == email.lower().strip())).scalar_one_or_none()
        if not u or not u.is_active:
            return None
        if not _verify_password(password, u.password_hash):
            return None
        return {"id": u.id, "email": u.email, "name": u.name, "role": u.role, "is_active": u.is_active}


def list_users() -> List[Dict]:
    with get_session() as s:
        rows = s.execute(select(User).order_by(User.created_at.desc())).scalars().all()
        return [
            {"id": r.id, "email": r.email, "name": r.name, "role": r.role, "is_active": r.is_active, "created_at": r.created_at}
            for r in rows
        ]


def list_groups() -> List[Dict]:
    with get_session() as s:
        rows = s.execute(select(Group).order_by(Group.name.asc())).scalars().all()
        return [{"id": r.id, "name": r.name} for r in rows]


def create_group(name: str) -> Tuple[bool, str]:
    name = name.strip()
    if not name:
        return False, "Invalid name"
    with get_session() as s:
        ex = s.execute(select(Group).where(func.lower(Group.name) == name.lower())).scalar_one_or_none()
        if ex:
            return False, "Group already exists"
        g = Group(name=name)
        s.add(g)
        s.commit()
        return True, "Group created"


def delete_group(name: str) -> Tuple[bool, str]:
    with get_session() as s:
        g = s.execute(select(Group).where(func.lower(Group.name) == name.lower().strip())).scalar_one_or_none()
        if not g:
            return False, "Group not found"
        s.delete(g)
        s.commit()
        return True, "Group deleted"


def assign_user_to_group(user_email: str, group_name: str) -> Tuple[bool, str]:
    with get_session() as s:
        u = s.execute(select(User).where(User.email == user_email.lower().strip())).scalar_one_or_none()
        g = s.execute(select(Group).where(func.lower(Group.name) == group_name.lower().strip())).scalar_one_or_none()
        if not u or not g:
            return False, "User or Group not found"
        ex = s.execute(select(UserGroup).where(UserGroup.user_id == u.id, UserGroup.group_id == g.id)).scalar_one_or_none()
        if ex:
            return True, "Already assigned"
        s.add(UserGroup(user_id=u.id, group_id=g.id))
        s.commit()
        return True, "Assigned"


def unassign_user_from_group(user_email: str, group_name: str) -> Tuple[bool, str]:
    with get_session() as s:
        u = s.execute(select(User).where(User.email == user_email.lower().strip())).scalar_one_or_none()
        g = s.execute(select(Group).where(func.lower(Group.name) == group_name.lower().strip())).scalar_one_or_none()
        if not u or not g:
            return False, "User or Group not found"
        link = s.execute(select(UserGroup).where(UserGroup.user_id == u.id, UserGroup.group_id == g.id)).scalar_one_or_none()
        if not link:
            return False, "Not a member"
        s.delete(link)
        s.commit()
        return True, "Unassigned"


def list_visualizations() -> List[Dict]:
    with get_session() as s:
        rows = s.execute(select(Visualization).order_by(Visualization.name.asc())).scalars().all()
        return [{"id": r.id, "key": r.key, "name": r.name, "is_active": r.is_active} for r in rows]


def set_visualization_active(key: str, active: bool) -> Tuple[bool, str]:
    with get_session() as s:
        v = s.execute(select(Visualization).where(Visualization.key == key)).scalar_one_or_none()
        if not v:
            return False, "Visualization not found"
        v.is_active = active
        s.commit()
        return True, "Updated"


def allow_viz_for_user(user_email: str, viz_key: str, allowed: bool) -> Tuple[bool, str]:
    with get_session() as s:
        u = s.execute(select(User).where(User.email == user_email.lower().strip())).scalar_one_or_none()
        v = s.execute(select(Visualization).where(Visualization.key == viz_key)).scalar_one_or_none()
        if not u or not v:
            return False, "User or Visualization not found"
        bind = s.execute(select(UserVisualization).where(UserVisualization.user_id == u.id, UserVisualization.viz_id == v.id)).scalar_one_or_none()
        if not bind:
            bind = UserVisualization(user_id=u.id, viz_id=v.id, allowed=allowed)
            s.add(bind)
        else:
            bind.allowed = allowed
        s.commit()
        return True, "Updated"


def allow_viz_for_group(group_name: str, viz_key: str, allowed: bool) -> Tuple[bool, str]:
    with get_session() as s:
        g = s.execute(select(Group).where(func.lower(Group.name) == group_name.lower().strip())).scalar_one_or_none()
        v = s.execute(select(Visualization).where(Visualization.key == viz_key)).scalar_one_or_none()
        if not g or not v:
            return False, "Group or Visualization not found"
        bind = s.execute(select(GroupVisualization).where(GroupVisualization.group_id == g.id, GroupVisualization.viz_id == v.id)).scalar_one_or_none()
        if not bind:
            bind = GroupVisualization(group_id=g.id, viz_id=v.id, allowed=allowed)
            s.add(bind)
        else:
            bind.allowed = allowed
        s.commit()
        return True, "Updated"


def _collect_allowed_viz_for_user_id(s: Session, user_id: int) -> Set[str]:
    # Start with all active visualizations
    all_active = {k for (k,) in s.execute(select(Visualization.key).where(Visualization.is_active == True)).all()}  # noqa: E712

    # Explicit user overrides
    rows = s.execute(
        select(Visualization.key, UserVisualization.allowed)
        .join(UserVisualization, UserVisualization.viz_id == Visualization.id)
        .where(UserVisualization.user_id == user_id)
    ).all()
    user_map: Dict[str, bool] = {key: allowed for key, allowed in rows}

    # Group-level grants
    group_rows = s.execute(
        select(Visualization.key, GroupVisualization.allowed)
        .join(GroupVisualization, GroupVisualization.viz_id == Visualization.id)
        .join(UserGroup, UserGroup.group_id == GroupVisualization.group_id)
        .where(UserGroup.user_id == user_id)
    ).all()
    group_map: Dict[str, bool] = {}
    for key, allowed in group_rows:
        # Last-one-wins policy across groups; explicit user override outranks later
        group_map[key] = allowed

    # Merge precedence: user_map > group_map > default active
    allowed_keys: Set[str] = set()
    for key in all_active:
        val = True
        if key in group_map:
            val = group_map[key]
        if key in user_map:
            val = user_map[key]
        if val:
            allowed_keys.add(key)
    return allowed_keys


def allowed_visualizations_for(email: str) -> Set[str]:
    with get_session() as s:
        u = s.execute(select(User).where(User.email == email.lower().strip())).scalar_one_or_none()
        if not u:
            return set()
        if u.role == "admin":
            # Admin sees all active
            return {k for (k,) in s.execute(select(Visualization.key).where(Visualization.is_active == True)).all()}  # noqa: E712
        return _collect_allowed_viz_for_user_id(s, u.id)


def create_password_reset(email: str, ttl_minutes: int = 30) -> Tuple[bool, str]:
    with get_session() as s:
        u = s.execute(select(User).where(User.email == email.lower().strip())).scalar_one_or_none()
        if not u:
            return False, "User not found"
        token = secrets.token_urlsafe(32)
        expires = datetime.utcnow() + timedelta(minutes=ttl_minutes)
        pr = PasswordResetToken(user_id=u.id, token=token, expires_at=expires, used=False)
        s.add(pr)
        s.commit()
        # Try email if configured
        try:
            send_reset_email(u.email, token)
        except Exception:
            # Swallow to not fail the flow; admin or user can use the token manually
            pass
        return True, token


def reset_password_with_token(token: str, new_password: str) -> Tuple[bool, str]:
    with get_session() as s:
        pr = s.execute(select(PasswordResetToken).where(PasswordResetToken.token == token)).scalar_one_or_none()
        if not pr:
            return False, "Invalid token"
        if pr.used:
            return False, "Token already used"
        if pr.expires_at < datetime.utcnow():
            return False, "Token expired"
        u = s.execute(select(User).where(User.id == pr.user_id)).scalar_one_or_none()
        if not u:
            return False, "User not found"
        u.password_hash = _hash_password(new_password)
        pr.used = True
        s.commit()
        return True, "Password reset"


def _smtp_config() -> Optional[Dict]:
    cfg = {}
    if st is not None:
        try:
            for k in ["SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM", "SMTP_TLS"]:
                if k in st.secrets:
                    cfg[k] = st.secrets[k]
        except Exception:
            pass
    # Minimal required
    if not cfg.get("SMTP_HOST") or not cfg.get("SMTP_FROM"):
        return None
    # Defaults
    cfg.setdefault("SMTP_PORT", 587)
    cfg.setdefault("SMTP_TLS", True)
    return cfg


def send_reset_email(to_email: str, token: str) -> None:
    cfg = _smtp_config()
    if not cfg:
        raise RuntimeError("SMTP not configured in secrets; cannot send email.")

    subject = "Password Reset Instructions"
    # Provide app URL hint if present
    app_url = None
    if st is not None:
        try:
            app_url = st.secrets.get("APP_URL")
        except Exception:
            pass
    body = [
        "Hello,",
        "",
        "You (or an admin) requested a password reset.",
        f"Reset token: {token}",
        "",
        "Instructions:",
        "1) Open the app",
        "2) Click 'Reset Password' on the login screen",
        "3) Paste the token and choose a new password",
    ]
    if app_url:
        body.insert(2, f"Open: {app_url}")

    msg = EmailMessage()
    msg["From"] = str(cfg["SMTP_FROM"])
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content("\n".join(body))

    host = str(cfg["SMTP_HOST"]) 
    port = int(cfg.get("SMTP_PORT", 587))
    user = str(cfg.get("SMTP_USER", ""))
    password = str(cfg.get("SMTP_PASSWORD", ""))
    use_tls = bool(cfg.get("SMTP_TLS", True))

    if use_tls:
        with smtplib.SMTP(host, port) as smtp:
            smtp.starttls()
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)
    else:
        with smtplib.SMTP(host, port) as smtp:
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)

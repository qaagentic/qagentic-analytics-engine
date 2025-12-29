"""Report sharing and distribution system."""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


class SharePermission(str, Enum):
    """Share permissions."""
    VIEW = "view"
    COMMENT = "comment"
    EDIT = "edit"
    ADMIN = "admin"


class ShareType(str, Enum):
    """Types of shares."""
    PUBLIC = "public"
    LINK = "link"
    USER = "user"
    TEAM = "team"
    ORGANIZATION = "organization"


class DistributionChannel(str, Enum):
    """Distribution channels."""
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"


@dataclass
class ShareRecipient:
    """A recipient of a shared report."""
    recipient_id: str
    recipient_name: str
    recipient_type: str  # user, team, organization
    permission: SharePermission
    shared_at: datetime
    accessed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShareLink:
    """A shareable link for a report."""
    link_id: str
    report_id: str
    link_token: str
    permission: SharePermission
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportShare:
    """A report share record."""
    share_id: str
    report_id: str
    share_type: ShareType
    recipients: List[ShareRecipient] = field(default_factory=list)
    share_links: List[ShareLink] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledDistribution:
    """A scheduled report distribution."""
    distribution_id: str
    report_id: str
    recipients: List[str]
    channels: List[DistributionChannel]
    schedule: str  # cron expression
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_sent_at: Optional[datetime] = None
    next_send_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportSharingService:
    """Service for sharing and distributing reports."""

    def __init__(self):
        """Initialize report sharing service."""
        self.shares: Dict[str, ReportShare] = {}
        self.share_links: Dict[str, ShareLink] = {}
        self.distributions: Dict[str, ScheduledDistribution] = {}
        self.access_logs: List[Dict[str, Any]] = []

    def share_report(
        self,
        report_id: str,
        share_type: ShareType,
        recipients: List[str],
        permission: SharePermission = SharePermission.VIEW,
        created_by: str = "system",
    ) -> ReportShare:
        """
        Share a report with recipients.

        Args:
            report_id: ID of the report
            share_type: Type of share
            recipients: List of recipient IDs
            permission: Permission level
            created_by: User who created the share

        Returns:
            ReportShare object
        """
        share_id = f"share-{uuid.uuid4()}"

        share_recipients = [
            ShareRecipient(
                recipient_id=rid,
                recipient_name=rid,
                recipient_type="user",
                permission=permission,
                shared_at=datetime.utcnow(),
            )
            for rid in recipients
        ]

        share = ReportShare(
            share_id=share_id,
            report_id=report_id,
            share_type=share_type,
            recipients=share_recipients,
            created_by=created_by,
        )

        self.shares[share_id] = share
        logger.info(
            f"Shared report {report_id} with {len(recipients)} recipients "
            f"(share_type={share_type}, permission={permission})"
        )

        return share

    def create_share_link(
        self,
        report_id: str,
        permission: SharePermission = SharePermission.VIEW,
        expires_in_days: Optional[int] = None,
    ) -> ShareLink:
        """
        Create a shareable link for a report.

        Args:
            report_id: ID of the report
            permission: Permission level
            expires_in_days: Number of days until link expires

        Returns:
            ShareLink object
        """
        link_id = f"link-{uuid.uuid4()}"
        link_token = str(uuid.uuid4())

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        link = ShareLink(
            link_id=link_id,
            report_id=report_id,
            link_token=link_token,
            permission=permission,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
        )

        self.share_links[link_id] = link
        logger.info(
            f"Created share link for report {report_id}: {link_token} "
            f"(expires={expires_at})"
        )

        return link

    def access_report_via_link(self, link_token: str) -> Optional[str]:
        """
        Access a report via share link.

        Args:
            link_token: Share link token

        Returns:
            Report ID if access granted, None otherwise
        """
        for link in self.share_links.values():
            if link.link_token == link_token:
                # Check if link is expired
                if link.expires_at and link.expires_at < datetime.utcnow():
                    logger.warning(f"Share link {link.link_id} has expired")
                    return None

                # Update access info
                link.access_count += 1
                link.last_accessed_at = datetime.utcnow()

                # Log access
                self.access_logs.append({
                    "link_id": link.link_id,
                    "report_id": link.report_id,
                    "accessed_at": datetime.utcnow(),
                    "access_count": link.access_count,
                })

                logger.info(f"Report {link.report_id} accessed via link {link.link_id}")
                return link.report_id

        return None

    def revoke_share(self, share_id: str) -> bool:
        """
        Revoke a report share.

        Args:
            share_id: ID of the share

        Returns:
            True if revoked, False otherwise
        """
        if share_id not in self.shares:
            logger.warning(f"Share {share_id} not found")
            return False

        del self.shares[share_id]
        logger.info(f"Revoked share {share_id}")
        return True

    def revoke_share_link(self, link_id: str) -> bool:
        """
        Revoke a share link.

        Args:
            link_id: ID of the link

        Returns:
            True if revoked, False otherwise
        """
        if link_id not in self.share_links:
            logger.warning(f"Share link {link_id} not found")
            return False

        del self.share_links[link_id]
        logger.info(f"Revoked share link {link_id}")
        return True

    def schedule_distribution(
        self,
        report_id: str,
        recipients: List[str],
        channels: List[DistributionChannel],
        schedule: str,
        created_by: str = "system",
    ) -> ScheduledDistribution:
        """
        Schedule a report for distribution.

        Args:
            report_id: ID of the report
            recipients: List of recipient IDs
            channels: Distribution channels
            schedule: Cron expression for schedule
            created_by: User who created the distribution

        Returns:
            ScheduledDistribution object
        """
        distribution_id = f"dist-{uuid.uuid4()}"

        distribution = ScheduledDistribution(
            distribution_id=distribution_id,
            report_id=report_id,
            recipients=recipients,
            channels=channels,
            schedule=schedule,
            created_at=datetime.utcnow(),
        )

        self.distributions[distribution_id] = distribution
        logger.info(
            f"Scheduled distribution for report {report_id}: "
            f"recipients={len(recipients)}, channels={channels}, schedule={schedule}"
        )

        return distribution

    def get_scheduled_distributions(self, report_id: str) -> List[ScheduledDistribution]:
        """
        Get scheduled distributions for a report.

        Args:
            report_id: ID of the report

        Returns:
            List of ScheduledDistribution objects
        """
        return [
            d for d in self.distributions.values()
            if d.report_id == report_id
        ]

    def update_distribution_schedule(
        self,
        distribution_id: str,
        schedule: str,
    ) -> Optional[ScheduledDistribution]:
        """
        Update a distribution schedule.

        Args:
            distribution_id: ID of the distribution
            schedule: New cron expression

        Returns:
            Updated ScheduledDistribution or None
        """
        distribution = self.distributions.get(distribution_id)
        if not distribution:
            logger.warning(f"Distribution {distribution_id} not found")
            return None

        distribution.schedule = schedule
        logger.info(f"Updated distribution {distribution_id} schedule to {schedule}")
        return distribution

    def disable_distribution(self, distribution_id: str) -> bool:
        """
        Disable a scheduled distribution.

        Args:
            distribution_id: ID of the distribution

        Returns:
            True if disabled, False otherwise
        """
        distribution = self.distributions.get(distribution_id)
        if not distribution:
            logger.warning(f"Distribution {distribution_id} not found")
            return False

        distribution.enabled = False
        logger.info(f"Disabled distribution {distribution_id}")
        return True

    def get_shares_for_report(self, report_id: str) -> List[ReportShare]:
        """
        Get all shares for a report.

        Args:
            report_id: ID of the report

        Returns:
            List of ReportShare objects
        """
        return [
            share for share in self.shares.values()
            if share.report_id == report_id
        ]

    def get_shares_for_user(self, user_id: str) -> List[ReportShare]:
        """
        Get all shares for a user.

        Args:
            user_id: ID of the user

        Returns:
            List of ReportShare objects
        """
        shares = []
        for share in self.shares.values():
            for recipient in share.recipients:
                if recipient.recipient_id == user_id:
                    shares.append(share)
                    break
        return shares

    def update_recipient_permission(
        self,
        share_id: str,
        recipient_id: str,
        new_permission: SharePermission,
    ) -> bool:
        """
        Update a recipient's permission.

        Args:
            share_id: ID of the share
            recipient_id: ID of the recipient
            new_permission: New permission level

        Returns:
            True if updated, False otherwise
        """
        share = self.shares.get(share_id)
        if not share:
            logger.warning(f"Share {share_id} not found")
            return False

        for recipient in share.recipients:
            if recipient.recipient_id == recipient_id:
                recipient.permission = new_permission
                logger.info(
                    f"Updated permission for {recipient_id} in share {share_id} "
                    f"to {new_permission}"
                )
                return True

        logger.warning(f"Recipient {recipient_id} not found in share {share_id}")
        return False

    def get_access_logs(
        self,
        report_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get access logs.

        Args:
            report_id: Filter by report ID
            limit: Maximum number of logs

        Returns:
            List of access log entries
        """
        logs = self.access_logs

        if report_id:
            logs = [l for l in logs if l.get("report_id") == report_id]

        # Sort by accessed_at (most recent first)
        logs.sort(key=lambda l: l.get("accessed_at", datetime.utcnow()), reverse=True)

        return logs[:limit]

    def get_sharing_statistics(self) -> Dict[str, Any]:
        """
        Get sharing statistics.

        Returns:
            Dictionary with sharing statistics
        """
        return {
            "total_shares": len(self.shares),
            "total_share_links": len(self.share_links),
            "active_distributions": sum(1 for d in self.distributions.values() if d.enabled),
            "total_access_logs": len(self.access_logs),
            "by_share_type": {
                share_type.value: sum(1 for s in self.shares.values() if s.share_type == share_type)
                for share_type in ShareType
            },
            "by_permission": {
                perm.value: sum(
                    len([r for r in s.recipients if r.permission == perm])
                    for s in self.shares.values()
                )
                for perm in SharePermission
            },
        }

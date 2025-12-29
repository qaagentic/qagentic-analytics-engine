"""Report template system for generating customizable reports."""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class ReportType(str, Enum):
    """Types of reports."""
    SUMMARY = "summary"
    DETAILED = "detailed"
    TREND = "trend"
    COMPARISON = "comparison"
    EXECUTIVE = "executive"
    TECHNICAL = "technical"
    CUSTOM = "custom"


class ReportFormat(str, Enum):
    """Report output formats."""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    CSV = "csv"
    MARKDOWN = "markdown"
    EXCEL = "excel"


class SectionType(str, Enum):
    """Types of report sections."""
    HEADER = "header"
    SUMMARY = "summary"
    METRICS = "metrics"
    CHARTS = "charts"
    TABLES = "tables"
    TRENDS = "trends"
    FAILURES = "failures"
    RECOMMENDATIONS = "recommendations"
    FOOTER = "footer"


@dataclass
class ReportSection:
    """A section in a report template."""
    section_id: str
    section_type: SectionType
    title: str
    description: str
    enabled: bool = True
    order: int = 0
    configuration: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportTemplate:
    """A report template definition."""
    template_id: str
    name: str
    description: str
    report_type: ReportType
    format: ReportFormat
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: str = "system"
    is_default: bool = False
    is_public: bool = False


@dataclass
class ReportData:
    """Data for a generated report."""
    report_id: str
    template_id: str
    report_type: ReportType
    title: str
    generated_at: datetime
    time_range_start: Optional[datetime] = None
    time_range_end: Optional[datetime] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    sections_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ReportTemplateService:
    """Service for managing report templates."""

    def __init__(self):
        """Initialize report template service."""
        self.templates: Dict[str, ReportTemplate] = {}
        self.generated_reports: Dict[str, ReportData] = {}
        self._initialize_default_templates()

    def _initialize_default_templates(self) -> None:
        """Initialize default report templates."""
        # Executive Summary Template
        executive_template = ReportTemplate(
            template_id="template-executive",
            name="Executive Summary",
            description="High-level overview for stakeholders",
            report_type=ReportType.EXECUTIVE,
            format=ReportFormat.HTML,
            is_default=True,
            is_public=True,
            sections=[
                ReportSection(
                    section_id="header",
                    section_type=SectionType.HEADER,
                    title="Report Header",
                    description="Title and metadata",
                    order=1,
                ),
                ReportSection(
                    section_id="summary",
                    section_type=SectionType.SUMMARY,
                    title="Executive Summary",
                    description="Key metrics and findings",
                    order=2,
                    configuration={"show_trends": True, "show_comparisons": True},
                ),
                ReportSection(
                    section_id="metrics",
                    section_type=SectionType.METRICS,
                    title="Key Metrics",
                    description="Pass rate, failure rate, flaky tests",
                    order=3,
                    configuration={"metrics": ["pass_rate", "failure_rate", "flaky_rate"]},
                ),
                ReportSection(
                    section_id="recommendations",
                    section_type=SectionType.RECOMMENDATIONS,
                    title="Recommendations",
                    description="Action items and improvements",
                    order=4,
                ),
            ],
        )
        self.templates[executive_template.template_id] = executive_template

        # Technical Details Template
        technical_template = ReportTemplate(
            template_id="template-technical",
            name="Technical Details",
            description="Detailed technical analysis for engineers",
            report_type=ReportType.TECHNICAL,
            format=ReportFormat.HTML,
            is_default=True,
            is_public=True,
            sections=[
                ReportSection(
                    section_id="header",
                    section_type=SectionType.HEADER,
                    title="Report Header",
                    description="Title and metadata",
                    order=1,
                ),
                ReportSection(
                    section_id="summary",
                    section_type=SectionType.SUMMARY,
                    title="Test Summary",
                    description="Overall test results",
                    order=2,
                ),
                ReportSection(
                    section_id="metrics",
                    section_type=SectionType.METRICS,
                    title="Detailed Metrics",
                    description="All metrics by service and branch",
                    order=3,
                    configuration={"granular": True, "by_service": True, "by_branch": True},
                ),
                ReportSection(
                    section_id="failures",
                    section_type=SectionType.FAILURES,
                    title="Top Failing Tests",
                    description="Tests with highest failure rates",
                    order=4,
                    configuration={"limit": 20, "show_stack_traces": True},
                ),
                ReportSection(
                    section_id="trends",
                    section_type=SectionType.TRENDS,
                    title="Trend Analysis",
                    description="Historical trends and patterns",
                    order=5,
                ),
                ReportSection(
                    section_id="recommendations",
                    section_type=SectionType.RECOMMENDATIONS,
                    title="Technical Recommendations",
                    description="Specific fixes and improvements",
                    order=6,
                ),
            ],
        )
        self.templates[technical_template.template_id] = technical_template

        # Trend Analysis Template
        trend_template = ReportTemplate(
            template_id="template-trend",
            name="Trend Analysis",
            description="Historical trends and patterns",
            report_type=ReportType.TREND,
            format=ReportFormat.HTML,
            is_default=True,
            is_public=True,
            sections=[
                ReportSection(
                    section_id="header",
                    section_type=SectionType.HEADER,
                    title="Report Header",
                    description="Title and metadata",
                    order=1,
                ),
                ReportSection(
                    section_id="trends",
                    section_type=SectionType.TRENDS,
                    title="Trend Analysis",
                    description="Pass rate, failure rate, and flaky test trends",
                    order=2,
                    configuration={"metrics": ["pass_rate", "failure_rate", "flaky_rate"], "interval": "day"},
                ),
                ReportSection(
                    section_id="charts",
                    section_type=SectionType.CHARTS,
                    title="Trend Charts",
                    description="Visual representation of trends",
                    order=3,
                    configuration={"chart_types": ["line", "area"], "show_forecast": True},
                ),
            ],
        )
        self.templates[trend_template.template_id] = trend_template

        logger.info(f"Initialized {len(self.templates)} default report templates")

    def create_template(
        self,
        name: str,
        description: str,
        report_type: ReportType,
        format: ReportFormat,
        sections: List[ReportSection],
        created_by: str = "system",
        is_public: bool = False,
    ) -> ReportTemplate:
        """
        Create a new report template.

        Args:
            name: Template name
            description: Template description
            report_type: Type of report
            format: Output format
            sections: List of report sections
            created_by: User who created the template
            is_public: Whether template is public

        Returns:
            Created ReportTemplate
        """
        template_id = f"template-{datetime.utcnow().timestamp()}"

        template = ReportTemplate(
            template_id=template_id,
            name=name,
            description=description,
            report_type=report_type,
            format=format,
            sections=sections,
            created_by=created_by,
            is_public=is_public,
        )

        self.templates[template_id] = template
        logger.info(f"Created report template: {template_id} ({name})")

        return template

    def get_template(self, template_id: str) -> Optional[ReportTemplate]:
        """
        Get a report template by ID.

        Args:
            template_id: ID of the template

        Returns:
            ReportTemplate or None
        """
        return self.templates.get(template_id)

    def list_templates(
        self,
        report_type: Optional[ReportType] = None,
        format: Optional[ReportFormat] = None,
        public_only: bool = False,
    ) -> List[ReportTemplate]:
        """
        List report templates.

        Args:
            report_type: Filter by report type
            format: Filter by format
            public_only: Only return public templates

        Returns:
            List of templates
        """
        templates = list(self.templates.values())

        if report_type:
            templates = [t for t in templates if t.report_type == report_type]

        if format:
            templates = [t for t in templates if t.format == format]

        if public_only:
            templates = [t for t in templates if t.is_public]

        return templates

    def update_template(
        self,
        template_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        sections: Optional[List[ReportSection]] = None,
    ) -> Optional[ReportTemplate]:
        """
        Update a report template.

        Args:
            template_id: ID of the template
            name: New name
            description: New description
            sections: New sections

        Returns:
            Updated ReportTemplate or None
        """
        template = self.templates.get(template_id)
        if not template:
            logger.warning(f"Template {template_id} not found")
            return None

        if name:
            template.name = name
        if description:
            template.description = description
        if sections:
            template.sections = sections

        template.updated_at = datetime.utcnow()
        logger.info(f"Updated report template: {template_id}")

        return template

    def delete_template(self, template_id: str) -> bool:
        """
        Delete a report template.

        Args:
            template_id: ID of the template

        Returns:
            True if deleted, False otherwise
        """
        if template_id not in self.templates:
            logger.warning(f"Template {template_id} not found")
            return False

        del self.templates[template_id]
        logger.info(f"Deleted report template: {template_id}")
        return True

    def generate_report(
        self,
        template_id: str,
        title: str,
        time_range_start: Optional[datetime] = None,
        time_range_end: Optional[datetime] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Optional[ReportData]:
        """
        Generate a report from a template.

        Args:
            template_id: ID of the template to use
            title: Report title
            time_range_start: Start of time range
            time_range_end: End of time range
            filters: Optional filters

        Returns:
            Generated ReportData or None
        """
        template = self.templates.get(template_id)
        if not template:
            logger.warning(f"Template {template_id} not found")
            return None

        report_id = f"report-{datetime.utcnow().timestamp()}"

        report = ReportData(
            report_id=report_id,
            template_id=template_id,
            report_type=template.report_type,
            title=title,
            generated_at=datetime.utcnow(),
            time_range_start=time_range_start,
            time_range_end=time_range_end,
            filters=filters or {},
        )

        self.generated_reports[report_id] = report
        logger.info(f"Generated report: {report_id} from template {template_id}")

        return report

    def get_report(self, report_id: str) -> Optional[ReportData]:
        """
        Get a generated report.

        Args:
            report_id: ID of the report

        Returns:
            ReportData or None
        """
        return self.generated_reports.get(report_id)

    def list_reports(
        self,
        template_id: Optional[str] = None,
        report_type: Optional[ReportType] = None,
        limit: int = 50,
    ) -> List[ReportData]:
        """
        List generated reports.

        Args:
            template_id: Filter by template
            report_type: Filter by report type
            limit: Maximum number of reports

        Returns:
            List of reports
        """
        reports = list(self.generated_reports.values())

        if template_id:
            reports = [r for r in reports if r.template_id == template_id]

        if report_type:
            reports = [r for r in reports if r.report_type == report_type]

        # Sort by generated_at (most recent first)
        reports.sort(key=lambda r: r.generated_at, reverse=True)

        return reports[:limit]

    def export_report(
        self,
        report_id: str,
        format: ReportFormat,
    ) -> Optional[str]:
        """
        Export a report in a specific format.

        Args:
            report_id: ID of the report
            format: Export format

        Returns:
            Exported report content or None
        """
        report = self.generated_reports.get(report_id)
        if not report:
            logger.warning(f"Report {report_id} not found")
            return None

        logger.info(f"Exporting report {report_id} as {format}")

        # In a real implementation, this would generate the actual report
        # For now, return a placeholder
        if format == ReportFormat.JSON:
            return json.dumps({
                "report_id": report.report_id,
                "title": report.title,
                "generated_at": report.generated_at.isoformat(),
                "sections": report.sections_data,
            })

        return None

    def get_template_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about templates.

        Returns:
            Dictionary with template statistics
        """
        templates = list(self.templates.values())
        reports = list(self.generated_reports.values())

        by_type = {}
        for report_type in ReportType:
            count = sum(1 for t in templates if t.report_type == report_type)
            if count > 0:
                by_type[report_type.value] = count

        by_format = {}
        for format in ReportFormat:
            count = sum(1 for t in templates if t.format == format)
            if count > 0:
                by_format[format.value] = count

        return {
            "total_templates": len(templates),
            "total_reports_generated": len(reports),
            "by_type": by_type,
            "by_format": by_format,
            "default_templates": sum(1 for t in templates if t.is_default),
            "public_templates": sum(1 for t in templates if t.is_public),
        }

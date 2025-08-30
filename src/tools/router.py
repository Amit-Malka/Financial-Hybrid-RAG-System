from ..config import Config
import logging
import re

def route_query(query: str) -> str:
    """Enhanced routing with numerical content detection."""
    logger = logging.getLogger("tools.router")
    query_lower = query.lower()

    # Enhanced table routing - check for numerical/financial content
    has_numbers = bool(re.search(r'\d+', query))
    has_financial_terms = any(keyword in query_lower for keyword in Config.TABLE_KEYWORDS)

    if has_financial_terms or (has_numbers and any(word in query_lower for word in
        ['rate', 'cost', 'revenue', 'percentage', '%', 'quarter', 'year'])):
        logger.debug("Routing to table_tool (enhanced detection)")
        return "table_tool"
    elif any(keyword in query_lower for keyword in Config.RISK_KEYWORDS):
        logger.debug("Routing to risk_tool")
        return "risk_tool"
    elif any(keyword in query_lower for keyword in Config.MDA_KEYWORDS):
        logger.debug("Routing to mda_tool")
        return "mda_tool"
    else:
        logger.debug("Routing to general_tool")
        return "general_tool"

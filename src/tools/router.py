from ..config import Config
import logging

def route_query(query: str) -> str:
    """Routes a query to the appropriate tool based on keywords."""
    logger = logging.getLogger("tools.router")
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in Config.TABLE_KEYWORDS):
        logger.debug("Routing to table_tool")
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

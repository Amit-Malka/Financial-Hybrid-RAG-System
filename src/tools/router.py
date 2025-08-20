from ..config import Config

def route_query(query: str) -> str:
    """Routes a query to the appropriate tool based on keywords."""
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in Config.TABLE_KEYWORDS):
        return "table_tool"
    elif any(keyword in query_lower for keyword in Config.RISK_KEYWORDS):
        return "risk_tool"
    elif any(keyword in query_lower for keyword in Config.MDA_KEYWORDS):
        return "mda_tool"
    else:
        return "general_tool"

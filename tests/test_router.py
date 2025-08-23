from src.tools.router import route_query


def test_route_query_keywords():
    assert route_query("show revenue table") == "table_tool"
    assert route_query("key risk factors in q2") == "risk_tool"
    assert route_query("management discussion and analysis outlook") == "mda_tool"
    assert route_query("tell me about the company") == "general_tool"



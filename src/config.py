class Config:
    # Core settings
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50
    
    # Hybrid retrieval (simplified)
    DENSE_WEIGHT = 0.8
    TFIDF_WEIGHT = 0.2
    
    # 10-Q specific TF-IDF terms
    FINANCIAL_10Q_TERMS = {
        # Core financial
        'revenue', 'assets', 'liabilities', 'equity', 'cash_flow',
        # 10-Q specific
        'quarterly', 'interim', 'unaudited', 'condensed',
        'yoy', 'quarter_over_quarter', 'guidance', 'outlook',
        # SEC specific
        'md_a', 'risk_factors', 'forward_looking', 'material'
    }
    
    # Simple TF-IDF settings
    MAX_FEATURES = 5000
    FINANCIAL_BOOST = 2.0
    
    # Tools routing keywords
    TABLE_KEYWORDS = ['revenue', 'income', 'balance', 'cash_flow', 'financial_statement']
    RISK_KEYWORDS = ['risk', 'uncertainty', 'factor', 'may_adversely']
    MDA_KEYWORDS = ['management', 'discussion', 'analysis', 'outlook', 'results']
    
    # UI settings
    DEFAULT_TOP_K = 5
    GOOGLE_API_KEY = ""  # User provided

    # Metadata schema
    METADATA_SCHEMA = {
        "section_path": [
            "part_i/item_1/financial_statements",
            "part_i/item_2/md_a", 
            "part_i/item_3/market_risk",
            "part_i/item_4/controls",
            "part_ii/item_1a/risk_factors",
            "part_ii/other"
        ],
        "content_type": [
            "balance_sheet", "income_statement", "cash_flow", 
            "notes", "md_a_narrative", "risk_disclosure", "legal"
        ],
        "page_number": int
    }

    # Logging configuration
    LOG_DIR = "logs"
    LOG_LEVEL = "DEBUG"  # One of: DEBUG, INFO, WARNING, ERROR
    LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
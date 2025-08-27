class Config:
    # Core settings
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 50
    
    # Hybrid retrieval (per specification: 70% Dense + 30% TF-IDF + 15% Graph Enhancement)
    DENSE_WEIGHT = 0.7
    TFIDF_WEIGHT = 0.3
    GRAPH_ENHANCEMENT_WEIGHT = 0.15
    
    # 10-Q specific TF-IDF terms
    FINANCIAL_10Q_TERMS = {
        # Core financial
        'revenue', 'assets', 'liabilities', 'equity', 'cash_flow',
        # 10-Q specific
        'quarterly', 'interim', 'unaudited', 'condensed',
        'yoy', 'quarter_over_quarter', 'guidance', 'outlook',
        # SEC specific
        'md_a', 'risk_factors', 'forward_looking', 'material',
        # Monetization metrics
        'cost_per_click', 'paid_clicks', 'impressions', 'monetization',
        'cost_per_impression', 'click_through_rate', 'advertising'
    }
    
    # Simple TF-IDF settings
    MAX_FEATURES = 5000
    FINANCIAL_BOOST = 2.0
    
    # Tools routing keywords
    TABLE_KEYWORDS = ['revenue', 'income', 'balance', 'cash_flow', 'financial_statement', 
                     'cost-per-click', 'cost_per_click', 'paid_clicks', 'paid clicks', 
                     'click', 'clicks', 'impressions', 'cost-per-impression', 'monetization']
    RISK_KEYWORDS = ['risk', 'uncertainty', 'factor', 'may_adversely']
    MDA_KEYWORDS = ['management', 'discussion', 'analysis', 'outlook', 'results']
    
    # UI settings
    DEFAULT_TOP_K = 5
    GOOGLE_API_KEY = ""  # User provided
    
    # Chunk ID configuration (5th metadata field)
    CHUNK_ID_PREFIX = "chunk_"
    
    # Graph enhancement settings
    ENABLE_GRAPH_ENHANCEMENT = True

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

    @classmethod
    def validate(cls):
        """Validate configuration values and constraints.

        - Dense + TF-IDF must sum to 1.0 (±0.01)
        - Graph enhancement weight must be within [0.0, 0.5]
        - Chunk sizes must be positive and overlap less than size
        """
        if abs(cls.DENSE_WEIGHT + cls.TFIDF_WEIGHT - 1.0) > 0.01:
            raise ValueError("DENSE_WEIGHT + TFIDF_WEIGHT must equal 1.0")
        if cls.GRAPH_ENHANCEMENT_WEIGHT < 0.0 or cls.GRAPH_ENHANCEMENT_WEIGHT > 0.5:
            raise ValueError("GRAPH_ENHANCEMENT_WEIGHT must be between 0.0 and 0.5")
        if cls.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        if cls.CHUNK_OVERLAP < 0 or cls.CHUNK_OVERLAP >= cls.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and < CHUNK_SIZE")
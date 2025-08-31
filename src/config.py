import os

# Disable ChromaDB telemetry completely - must be set before any ChromaDB imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False" 
os.environ["CHROMA_DISABLE_TELEMETRY"] = "True"

class Config:
    # Core settings (increased for better financial document processing)
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
        # Debt and financing terms (CRITICAL for debt retrieval)
        'debt', 'term_debt', 'borrowings', 'notes_payable', 'bonds',
        'long_term_debt', 'short_term_debt', 'current_liabilities', 'total_debt',
        'credit_facilities', 'loans', 'finance_lease', 'operating_lease',
        'lease_liabilities', 'principal_amount', 'maturity', 'interest_expense',
        'weighted_average', 'basis_point', 'fair_value', 'corporate_debt',
        # Balance sheet terms
        'balance_sheet', 'consolidated_balance', 'statement_financial_position',
        'current_assets', 'non_current', 'stockholders_equity', 'retained_earnings',
        # 10-Q specific
        'quarterly', 'interim', 'unaudited', 'condensed',
        'yoy', 'quarter_over_quarter', 'guidance', 'outlook',
        # SEC specific
        'md_a', 'risk_factors', 'forward_looking', 'material',
        # Monetization metrics
        'cost_per_click', 'paid_clicks', 'impressions', 'monetization',
        'cost_per_impression', 'click_through_rate', 'advertising',
        # Strategic partnerships and alliances
        'partnership', 'strategic_partnership', 'alliance', 'joint_venture',
        'collaboration', 'openai', 'funding_commitment', 'investment',
        'billion_commitment', 'strategic_alliance', 'partner', 'agreement'
    }
    
    # Enhanced TF-IDF settings for financial documents
    MAX_FEATURES = 8000  # Increased to accommodate bigrams and financial terms
    FINANCIAL_BOOST = 2.5  # Slightly higher boost for financial terms
    
    # Tools routing keywords
    TABLE_KEYWORDS = ['revenue', 'income', 'balance', 'cash_flow', 'financial_statement',
                     'cost-per-click', 'cost_per_click', 'paid_clicks', 'paid clicks',
                     'click', 'clicks', 'impressions', 'cost-per-impression', 'monetization']

    # Enhanced table processing keywords
    ENHANCED_TABLE_KEYWORDS = [
        "table", "revenue", "income", "cost", "TAC", "traffic acquisition cost",
        "cost-per-click", "paid clicks", "quarter", "Q1", "Q2", "Q3", "Q4",
        "financial", "balance sheet", "cash flow", "earnings", "margins",
        "year-over-year", "percentage", "billion", "million", "rate"
    ]

    # Combine with existing table keywords
    TABLE_KEYWORDS = TABLE_KEYWORDS + ENHANCED_TABLE_KEYWORDS
    RISK_KEYWORDS = ['risk', 'uncertainty', 'factor', 'may_adversely']
    MDA_KEYWORDS = ['management', 'discussion', 'analysis', 'outlook', 'results']
    
    # UI settings
    DEFAULT_TOP_K = 5
    GOOGLE_API_KEY = ""  # User provided
    
    # Chunk ID configuration (5th metadata field)
    CHUNK_ID_PREFIX = "chunk_"
    
    # Graph enhancement settings
    ENABLE_GRAPH_ENHANCEMENT = True

    # Chunking behavior
    # If True, use section-aware semantic chunking; otherwise, legacy 1:1 element wrapping
    USE_SECTION_AWARE_CHUNKING = True

    # Graph SIMILAR_TO enhancement
    ENABLE_SIMILAR_TO = False
    SIMILAR_TOP_N = 5
    SIMILARITY_THRESHOLD = 0.7

    # Metadata schema (5 fields)
    METADATA_SCHEMA = {
        "element_type": str,
        "chunk_id": str,
        "page_number": int,
        "section_path": str,
        "content_type": str,
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
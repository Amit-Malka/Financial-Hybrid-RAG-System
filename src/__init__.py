# Marker to make `src` a package for test imports

import os

# Disable ChromaDB telemetry globally before any imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False" 
os.environ["CHROMA_DISABLE_TELEMETRY"] = "True"


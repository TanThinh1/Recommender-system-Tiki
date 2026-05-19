# Có thể export các hàm chính để dùng từ ngoài
from .config import log, C_RESET, C_GREEN, C_RED, C_CYAN, C_BOLD, C_DIM
from .reporting import print_quality_report
from .db import get_db, close_db, bulk_upsert, insert_interactions_safe
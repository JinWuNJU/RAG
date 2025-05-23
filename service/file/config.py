# 文件服务相关配置常量

# 每个用户允许的最大文件数量
MAX_FILES_PER_USER = 10000  # 设置为10000个文件的上限

# 全局文件大小限制
DEFAULT_MAX_FILE_SIZE_MB = 50  # 默认50MB

# 默认允许的文件类型（如果未在API中指定）
DEFAULT_ALLOWED_FILE_TYPES = [
    # Office文档
    "application/msword",  # DOC
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # DOCX
    "application/vnd.ms-powerpoint",  # PPT
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # PPTX
    "application/vnd.ms-excel",  # XLS
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # XLSX
    "application/rtf",  # RTF
    
    # PDF
    "application/pdf",
    
    # 表格数据
    "text/csv",  # CSV
    "text/tab-separated-values",  # TSV
    
    # 标记语言
    "text/html",  # HTML
    "application/xml",  # XML
    "text/xml",
    
    # 纯文本类
    "text/plain",  # TXT
    "text/markdown",  # Markdown
    
    # 图片格式
    "image/png",  # PNG
    "image/jpg",  # JPG
    "image/jpeg",  # JPEG
    "image/tiff",  # TIFF
    "image/bmp",  # BMP
    "image/gif",  # GIF
    "image/x-icon",  # ICO
    "image/svg+xml"  # SVG
] 
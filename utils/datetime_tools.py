from datetime import datetime, timedelta

def get_beijing_time():
    """
    获取当前北京时间 (UTC+8)
    
    Returns:
        datetime: 当前北京时间
    """
    return datetime.utcnow() + timedelta(hours=8)

def to_timestamp_ms(dt):
    """
    将时间转换为毫秒时间戳（基于北京时间）
    
    Args:
        dt (datetime or int): 日期时间对象或时间戳
    
    Returns:
        int: 毫秒时间戳
    """
    if isinstance(dt, datetime):
        # 确保使用北京时间
        beijing_time = dt
        return int(beijing_time.timestamp() * 1000)
    return int(dt)  # 如果已经是时间戳，直接返回 
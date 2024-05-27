from datetime import datetime

# 获取当前时间
now = datetime.now()

# 将当前时间转换为字符串，格式为 YYYY-MM-DD HH:MM:SS
time_str = now.strftime("%Y-%m-%d_%H:%M:%S")

print(time_str)

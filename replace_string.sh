#!/bin/bash

#!/bin/bash

# 定义目标文件夹路径
TARGET_DIR="taut_src"

# 查找目标文件夹下所有的 .py 文件并替换指定内容
find "$TARGET_DIR" -type f -name "*.py" | while read -r file; do
  sed -i '' 's/moltaut_src/taut_src/g' "$file"
done

echo "替换完成。"


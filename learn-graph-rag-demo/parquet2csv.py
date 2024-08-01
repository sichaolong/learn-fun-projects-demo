import os
import pandas as pd

# 指定输入和输出文件夹路径
input_folder = './graphrag-local-ollama/ragtest/output/20240730-133354/artifacts'
output_folder = './graphrag-local-ollama/ragtest/output/20240730-133354/parquet2csvs'


def convert_parquet_to_csv(input_folder, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".parquet"):
            # 构建完整的文件路径
            parquet_file = os.path.join(input_folder, filename)
            csv_file = os.path.join(output_folder, filename.replace(".parquet", ".csv"))

            # 读取 Parquet 文件
            df = pd.read_parquet(parquet_file)

            # 将数据写入 CSV 文件
            df.to_csv(csv_file, index=False)

            print(f"Converted {parquet_file} to {csv_file}")


# 调用函数进行转换
convert_parquet_to_csv(input_folder, output_folder)

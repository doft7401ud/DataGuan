import pandas as pd
import os

top_folder_path = './'

# 遍历从No1到No39的所有文件夹
for i in range(1, 40):
    folder_path = os.path.join(top_folder_path, f'No{i}')
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 遍历每个文件夹中的所有文件
        filenames = os.listdir(folder_path)
        for filename in filenames:
            if 'movingGazeRatio' in filename:
                file_path = os.path.join(folder_path, filename)
                # 确保文件是Excel文件
                if file_path.endswith('.xlsx'):
                    # 使用ExcelFile类打开文件，以便读取所有工作表
                    with pd.ExcelFile(file_path) as xls:
                        # 创建一个空字典来存储转换后的工作表
                        transformed_sheets = {}
                        # 遍历所有工作表
                        for sheet_name in xls.sheet_names:
                            # 读取工作表，没有列名
                            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
                            # 转换Series为DataFrame
                            transformed_df = pd.DataFrame(df.iloc[0])
                            # 重新设置列名（如果需要）
                            # transformed_df.columns = ['Data']
                            # 将转换后的DataFrame存入字典
                            transformed_sheets[sheet_name] = transformed_df
                        # 定义新的文件路径进行保存
                        new_file_path = os.path.join(folder_path, f'transposed_{filename}')
                        with pd.ExcelWriter(new_file_path) as writer:
                            for sheet_name, data in transformed_sheets.items():
                                data.to_excel(writer, sheet_name=sheet_name, index=False)

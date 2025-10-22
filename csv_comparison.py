import csv
import sys


def compare_csv(file1_path, file2_path):
    """
    对比两个CSV文件的内容，返回所有不匹配的单元格信息

    参数:
        file1_path: 第一个CSV文件的路径
        file2_path: 第二个CSV文件的路径

    返回:
        包含所有差异的列表，每个差异是一个字典，包含行号、列号、列名、两个文件的值
    """
    differences = []

    try:
        with open(file1_path, 'r', newline='', encoding='utf-8') as f1, \
                open(file2_path, 'r', newline='', encoding='utf-8') as f2:

            reader1 = csv.DictReader(f1)
            reader2 = csv.DictReader(f2)

            # 检查列名是否一致
            if reader1.fieldnames != reader2.fieldnames:
                differences.append({
                    'type': 'column_mismatch',
                    'message': f"列名不匹配: {reader1.fieldnames} vs {reader2.fieldnames}"
                })
                return differences

            fieldnames = reader1.fieldnames

            # 逐行比较
            for row_num, (row1, row2) in enumerate(zip(reader1, reader2), start=2):  # 行号从2开始，因为表头是第1行
                # 比较每行的每个字段
                for col_num, col_name in enumerate(fieldnames, start=1):
                    value1 = row1.get(col_name, '')
                    value2 = row2.get(col_name, '')

                    # 尝试将值转换为数字进行比较，以便处理数值型数据的比较
                    try:
                        num1 = float(value1)
                        num2 = float(value2)
                        if not isclose(num1, num2):
                            differences.append({
                                'type': 'value_mismatch',
                                'row': row_num,
                                'col': col_num,
                                'col_name': col_name,
                                'file1_value': value1,
                                'file2_value': value2
                            })
                    except (ValueError, TypeError):
                        # 如果不能转换为数字，则直接比较字符串
                        if value1 != value2:
                            differences.append({
                                'type': 'value_mismatch',
                                'row': row_num,
                                'col': col_num,
                                'col_name': col_name,
                                'file1_value': value1,
                                'file2_value': value2
                            })

            # 检查文件行数是否一致
            # 尝试读取下一行，判断是否有剩余行
            try:
                next(reader1)
                differences.append({
                    'type': 'row_count_mismatch',
                    'message': f"{file1_path} 比 {file2_path} 包含更多行"
                })
            except StopIteration:
                pass

            try:
                next(reader2)
                differences.append({
                    'type': 'row_count_mismatch',
                    'message': f"{file2_path} 比 {file1_path} 包含更多行"
                })
            except StopIteration:
                pass

    except FileNotFoundError as e:
        differences.append({'type': 'file_error', 'message': f"文件未找到: {e.filename}"})
    except Exception as e:
        differences.append({'type': 'error', 'message': f"比较过程中发生错误: {str(e)}"})

    return differences


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """
    比较两个浮点数是否接近，处理浮点数精度问题
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def print_differences(differences, file1, file2):
    """打印所有差异信息"""
    if not differences:
        print(f"{file1} 和 {file2} 内容完全一致")
        return

    print(f"共发现 {len(differences)} 处差异:")
    print("----------------------------------------")

    for diff in differences:
        if diff['type'] == 'column_mismatch':
            print(f"错误: {diff['message']}")
        elif diff['type'] == 'row_count_mismatch':
            print(f"错误: {diff['message']}")
        elif diff['type'] == 'file_error':
            print(f"文件错误: {diff['message']}")
        elif diff['type'] == 'error':
            print(f"错误: {diff['message']}")
        elif diff['type'] == 'value_mismatch':
            print(f"行 {diff['row']}, 列 {diff['col']} ({diff['col_name']}):")
            print(f"  {file1}: {diff['file1_value']}")
            print(f"  {file2}: {diff['file2_value']}")
        print("----------------------------------------")


if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) != 3:
        print("用法: python csv_comparison.py <文件1路径> <文件2路径>")
        print("示例: python csv_comparison.py data1.csv data2.csv")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    # 比较CSV文件
    differences = compare_csv(file1, file2)

    # 打印差异结果
    print_differences(differences, file1, file2)

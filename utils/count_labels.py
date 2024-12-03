import json

def count_labels(data, max_participant=43):
    total_true = 0
    total_false = 0

    for participant_key in data:
        if participant_key.startswith("Participant_"):
            participant_num = int(participant_key.split("_")[1])
            if participant_num > max_participant:
                continue  # 只统计到 Participant_37

            experiments = data[participant_key]
            for experiment in experiments:
                label = experiment.get("label", False)
                if label:
                    total_true += 1
                else:
                    total_false += 1

    return total_true, total_false

if __name__ == "__main__":
    # 读取 JSON 数据
    with open(".\\label_1.json", "r") as file:
        data = json.load(file)

    true_count, false_count = count_labels(data)
    total = true_count + false_count

    if total > 0:
        true_ratio = true_count / total
        false_ratio = false_count / total
    else:
        true_ratio = false_ratio = 0

    print(f"True 总数: {true_count}")
    print(f"False 总数: {false_count}")
    print(f"True 比例: {true_ratio:.2%}")
    print(f"False 比例: {false_ratio:.2%}")

import os
import datetime
import torch
from train_and_evaluate import run_all_experiments, set_seed

def main():
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"=== 运行环境配置: 正在使用 {device} ===")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    dataset_folder = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "insects")
    )
    if not os.path.exists(dataset_folder):
        print("错误：未找到数据集目录")
        return

    epochs = 50
    
    print(f"\n>>> 开始运行干净数据实验 (Epochs={epochs})...")
    run_all_experiments(dataset_folder, False, output_dir, epochs=epochs)

    print(f"\n>>> 开始运行含噪数据实验 (Epochs={epochs})...")
    run_all_experiments(dataset_folder, True, output_dir, epochs=epochs)

    print(f"\n实验完成！结果已保存在 {output_dir}")

if __name__ == "__main__":
    main()
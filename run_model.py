import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
from matplotlib.font_manager import FontProperties

# --- Setup ---
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# --- Font Setup for Chinese Characters ---
def get_chinese_font():
    """Finds a Chinese font on the system."""
    try:
        # First, try to use 'SimHei' directly, a common Chinese font.
        return FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=12)
    except:
        # Fallback for other systems or if the font is not in the standard path
        try:
            return FontProperties(fname='SimHei.ttf', size=12)
        except:
            print("Warning: 'SimHei' or 'wqy-zenhei' font not found. Chinese characters in plots might not display correctly.")
            print("Please install a Chinese font (e.g., 'sudo apt-get install fonts-wqy-zenhei') and clear matplotlib cache.")
            return FontProperties(size=12) # Use default font

CHINESE_FONT = get_chinese_font()


# --- Step 3.4 Part 1: Transformer Model Definition ---
class TabularTransformer(nn.Module):
    """A Transformer-based model for tabular data classification."""
    def __init__(self, num_features, d_model=32, nhead=4, num_layers=2, num_classes=2):
        super(TabularTransformer, self).__init__()
        self.embedding = nn.Linear(num_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        out = self.classifier(x)
        return out

# --- Step 3.5: Updated Data Visualization ---
def visualize_results_updated(df_train, df_predictions, df_importance, df_merged, epoch):
    """
    Creates visualizations based on the user-provided images.
    Saves plots for a specific epoch.
    """
    print(f"\n--- Generating Visualizations for Epoch {epoch} ---")

    textprops = {"fontproperties": CHINESE_FONT}

    # --- Figure 1: Risk Proportions (2x2 Grid with Pies and Tables) ---
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
    
    # --- Top Row: Overall Dataset Composition ---
    ax1 = plt.subplot(gs[0, 0])
    # Corrected Logic: Show Audited vs. Unaudited users from the whole dataset
    overall_counts = df_merged['数据来源标识'].value_counts()
    overall_labels_map = {1: '已审核用户', 2: '待审核用户'}
    overall_labels = [overall_labels_map[i] for i in overall_counts.index]
    ax1.pie(overall_counts, labels=overall_labels, autopct='%1.1f%%', startangle=90, colors=['#4682B4', '#90EE90'] ,textprops=textprops)
    ax1.set_title('已审核和待审核用户对比', fontproperties=CHINESE_FONT, fontsize=16)

    ax2 = plt.subplot(gs[0, 1])
    ax2.axis('off')
    table_data_overall = [[overall_labels_map[i], count] for i, count in sorted(overall_counts.items())]
    table = ax2.table(cellText=table_data_overall, colLabels=['用户类型', '用户ID计数'], loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.2)
    for (row, col), cell in table.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=CHINESE_FONT, weight='bold')
        else:
            cell.set_text_props(fontproperties=CHINESE_FONT)

    # --- Bottom Row: Composition after Prediction ---
    ax3 = plt.subplot(gs[1, 0])
    # Corrected Logic: Show Audited (Non-compliant), Predicted Risk, Predicted No Risk
    pred_counts = df_predictions['预测分类'].value_counts()
    # Labels: 0 = Risk, 1 = No Risk
    count_pred_risk = pred_counts.get(0, 0)
    count_pred_no_risk = pred_counts.get(1, 0)
    count_audited = len(df_train) # Audited users are the 'non-compliant' part
    
    final_composition_counts = [count_pred_risk, count_pred_no_risk, count_audited]
    final_composition_labels = ['预测风险用户', '预测无风险用户', '不规范数据用户 (已审核)']
    
    # Filter out zero-count categories to avoid pie chart errors
    valid_labels = [label for i, label in enumerate(final_composition_labels) if final_composition_counts[i] > 0]
    valid_counts = [count for count in final_composition_counts if count > 0]

    ax3.pie(valid_counts, labels=valid_labels, autopct='%1.1f%%', startangle=90, colors=['#CD5C5C', '#8FBC8F', '#F0E68C'] ,textprops=textprops)
    ax3.set_title('预测风险、无风险及不规范数据用户对比', fontproperties=CHINESE_FONT, fontsize=16)

    ax4 = plt.subplot(gs[1, 1])
    ax4.axis('off')
    table_data_final = [
        ['预测风险用户', count_pred_risk],
        ['预测无风险用户', count_pred_no_risk],
        ['不规范数据用户 (已审核)', count_audited]
    ]
    table2 = ax4.table(cellText=table_data_final, colLabels=['用户类型', '用户ID计数'], loc='center', cellLoc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(12)
    table2.scale(1.2, 1.2)
    for (row, col), cell in table2.get_celld().items():
        if (row == 0):
            cell.set_text_props(fontproperties=CHINESE_FONT, weight='bold')
        else:
            cell.set_text_props(fontproperties=CHINESE_FONT)

    fig.suptitle(f'信贷风险比例展示 (Epoch {epoch})', fontproperties=CHINESE_FONT, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"credit_risk_proportions_epoch_{epoch}.png")
    plt.close(fig)


    # --- Figure 2: Key Indicators (Radar and Line Chart) ---
    fig2 = plt.figure(figsize=(16, 8))
    gs2 = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

    # Left: Radar chart for feature importance
    ax5 = plt.subplot(gs2[0], polar=True)
    labels = df_importance['指标'].values
    stats = df_importance['重要程度系数'].values
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    stats = np.concatenate((stats,[stats[0]]))
    angles = np.concatenate((angles,[angles[0]]))
    ax5.plot(angles, stats, 'o-', linewidth=2, color='deepskyblue')
    ax5.fill(angles, stats, alpha=0.25, color='deepskyblue')
    ax5.set_thetagrids(angles[:-1] * 180/np.pi, labels, fontproperties=CHINESE_FONT)
    ax5.set_title('关键指标展示', fontproperties=CHINESE_FONT, fontsize=16, y=1.1)
    ax5.grid(True)

    # Right: Line chart for key financial data of sample users
    ax6 = plt.subplot(gs2[1])
    sample_users = df_merged[df_merged['数据来源标识'] == 2].sample(n=20, random_state=42)
    user_ids = sample_users['用户ID']
    order_total = sample_users['订单总额']
    mean_val = sample_users['均值']
    std_val = sample_users['标准差']

    ax6.plot(user_ids, order_total, marker='o', label='订单总额', color='purple')
    ax6.plot(user_ids, mean_val, marker='s', label='均值', color='gold')
    
    ax6.set_title('重点指标数据展示', fontproperties=CHINESE_FONT, fontsize=16)
    ax6.set_xlabel('用户ID', fontproperties=CHINESE_FONT)
    ax6.set_ylabel('数值', fontproperties=CHINESE_FONT)
    plt.xticks(rotation=45, ha='right')
    
    ax6b = ax6.twinx()
    ax6b.plot(user_ids, std_val, marker='^', label='标准差', color='yellowgreen')
    ax6b.set_ylabel('标准差', fontproperties=CHINESE_FONT, color='yellowgreen')
    
    lines, labels = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6b.get_legend_handles_labels()
    ax6b.legend(lines + lines2, labels + labels2, loc=0, prop=CHINESE_FONT)


    fig2.suptitle(f'信贷风控关键指标展示 (Epoch {epoch})', fontproperties=CHINESE_FONT, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"credit_key_indicators_epoch_{epoch}.png")
    plt.close(fig2)


# --- Main Modeling and Prediction Function ---
def train_and_predict(file_path, max_epochs=5 ,lr=1e-4):
    """
    Loads merged data, trains the model, and saves results for each epoch.
    """
    print("\n--- Starting Model Training and Prediction ---")
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # --- Data Preparation ---
    df_train_orig = df[df['数据来源标识'] == 1].copy()
    df_predict_orig = df[df['数据来源标识'] == 2].copy()
    
    df_train_orig.dropna(subset=['审核标识'], inplace=True)
    df_train_orig['审核标识'] = df_train_orig['审核标识'].astype(int)

    feature_cols = [
        '认证间隔时间', '银行卡数量', '储蓄卡比例', '信用卡比例', '性别', '年龄',
        '订单总额', '均值', '标准差', '预留不同电话数量', '贷款次数'
    ]
    
    for col in feature_cols:
        median_val = df[col].median()
        df_train_orig[col].fillna(median_val, inplace=True)
        df_predict_orig[col].fillna(median_val, inplace=True)

    X = df_train_orig[feature_cols]
    y = df_train_orig['审核标识']
    X_pred_data = df_predict_orig[feature_cols]
    user_ids_pred = df_predict_orig['用户ID']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_pred_scaled = scaler.transform(X_pred_data)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train.values)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.LongTensor(y_val.values)
    X_pred_tensor = torch.FloatTensor(X_pred_scaled)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    # --- FIX: Define val_dataset before using it ---
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # --- Model Training ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TabularTransformer(num_features=len(feature_cols)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    train_losses = []
    
    print(f"Training on {device} for {max_epochs} epochs...")
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)

        # --- Validation and Result Generation for Current Epoch ---
        model.eval()
        
        # 1. Get predictions on validation set for importance calculation
        val_preds_list = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                outputs = model(batch_X.to(device))
                val_preds_list.extend(outputs.argmax(dim=1).cpu().numpy())
        accuracy = accuracy_score(y_val, val_preds_list)
        print(f"Epoch {epoch}/{max_epochs}, Training Loss: {avg_epoch_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
        
        # 2. Get predictions on actual application data
        with torch.no_grad():
            pred_outputs = model(X_pred_tensor.to(device))
            predictions = pred_outputs.argmax(dim=1).cpu().numpy()
        df_predictions = pd.DataFrame({'用户ID': user_ids_pred, '预测分类': predictions})
        df_predictions.to_csv(f'credit_classification_predictions_epoch_{epoch}.csv', index=False)

        # 3. Calculate Feature Importance
        importances = {}
        baseline_accuracy = accuracy_score(y_val, val_preds_list)
        with torch.no_grad():
            for i, col_name in enumerate(feature_cols):
                X_val_permuted = X_val_scaled.copy()
                np.random.shuffle(X_val_permuted[:, i])
                permuted_outputs = model(torch.FloatTensor(X_val_permuted).to(device))
                permuted_preds = permuted_outputs.argmax(dim=1).cpu().numpy()
                permuted_accuracy = accuracy_score(y_val, permuted_preds)
                importances[col_name] = baseline_accuracy - permuted_accuracy
        df_importance = pd.DataFrame(list(importances.items()), columns=['指标', '重要程度系数'])
        df_importance = df_importance.sort_values(by='重要程度系数', ascending=False)
        df_importance.to_csv(f'credit_feature_importance_epoch_{epoch}.csv', index=False)

        # 4. Generate Visualizations for this epoch
        visualize_results_updated(df_train_orig, df_predictions, df_importance, df, epoch)

    # --- Final Loss Curve Visualization ---
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_epochs + 1), train_losses, marker='o', linestyle='-')
    plt.title('模型训练损失曲线', fontproperties=CHINESE_FONT, fontsize=16)
    plt.xlabel('Epoch', fontproperties=CHINESE_FONT)
    plt.ylabel('平均损失 (Average Loss)', fontproperties=CHINESE_FONT)
    plt.xticks(range(1, max_epochs + 1))
    plt.grid(True)
    plt.savefig("training_loss_curve.png")
    plt.close()


# --- Main Execution Block ---
if __name__ == "__main__":
    # The input file is the one created by your merging script
    input_file = "merged_financial_data.csv"
    
    # Run the entire modeling and prediction pipeline
    train_and_predict(input_file, max_epochs=10 ,lr=2e-4)
    
    print("\nAnalysis complete. All results saved.")
    print("Check the main directory for the loss curve and epoch-specific results.")


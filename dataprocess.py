import pandas as pd
import re
# 读取Excel文件
df = pd.read_excel('uniport.xlsx')
#1生成fasta文件：
df = df.dropna(how='any')
df.to_excel('NotNullUniport.xlsx', index=False)

df['Sequence'] = df['Sequence'].apply(lambda seq: '\n'.join([seq[i:i+60] for i in range(0, len(seq), 60)]))

df['Entry'] = '>' + df['Entry']

fasta_data = df['Entry'] + '\n' + df['Sequence']
fasta_data.to_csv('output.fasta', sep='\n', header=False, index=False)

#2生成bp,mf,cc文件
# 创建空的DataFrame以保存结果
output_bp = pd.DataFrame(columns=['Entry', 'Gene Ontology (biological process)'])
output_mf = pd.DataFrame(columns=['Entry', 'Gene Ontology (molecular function)'])
output_cc = pd.DataFrame(columns=['Entry', 'Gene Ontology (cellular component)'])

# 遍历每一行
for index, row in df.iterrows():
    entry = row['Entry']
    bpo = row['Gene Ontology (biological process)']
    mfo = row['Gene Ontology (molecular function)']
    cco = row['Gene Ontology (cellular component)']
    go_terms_bp = re.findall(r'\[GO:(\d+)\]', bpo)
    go_terms_mf = re.findall(r'\[GO:(\d+)\]', mfo)
    go_terms_cc = re.findall(r'\[GO:(\d+)\]', cco)

    # 将每个GO项添加为新的行
    for go_term_bp in go_terms_bp:
        output_bp = output_bp.append({'Entry': entry, 'Gene Ontology (biological process)': go_term_bp}, ignore_index=True)
    for go_term_mf in go_terms_mf:
        output_mf = output_mf.append({'Entry': entry, 'Gene Ontology (molecular function)': go_term_mf}, ignore_index=True)
    for go_term_cc in go_terms_cc:
        output_cc = output_cc.append({'Entry': entry, 'Gene Ontology (cellular component)': go_term_cc}, ignore_index=True)

# 保存为output_gt_bp.txt文件
output_bp.to_csv('output_gt_bp.txt', sep='\t', index=False)
# 保存为output_gt_bp.txt文件
output_mf.to_csv('output_gt_mf.txt', sep='\t', index=False)
output_cc.to_csv('output_gt_cc.txt', sep='\t', index=False)

# 3提取第一列并保存为pid_list.txt
pid_list = df['Entry']
pid_list.to_csv('pid_list.txt', index=False)
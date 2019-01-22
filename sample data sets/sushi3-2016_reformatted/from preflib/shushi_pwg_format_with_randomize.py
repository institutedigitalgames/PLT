import pandas as pd

dir = "C:\\Users\\Beth\\PycharmProjects\\PyPLT\\sample data sets\\sushi3-2016_reformatted\\from preflib\\"

orig_a_order_path = dir+"sushi3_preflib-2016-pwg-a_order.csv"
orig_b_order_path = dir+"sushi3_preflib-2016-pwg-b_order.csv"
orig_b_score_path = dir+"sushi3_preflib-2016-pwg-b_score.csv"

orig_a_order_df = pd.read_csv(orig_a_order_path, header='infer')
orig_b_order_df = pd.read_csv(orig_b_order_path, header='infer')
orig_b_score_df = pd.read_csv(orig_b_score_path, header='infer')

# print(orig_a_order_df)
# print(orig_b_order_df)
# print(orig_b_score_df)

col_names = ["pref", "non_pref"]
new_paths = ["a_order_ranks", "b_order_ranks", "b_score_ranks"]
orig_dfs = [orig_a_order_df, orig_b_order_df, orig_b_score_df]

for file in range(len(orig_dfs)):
    df = []
    for idx, row in orig_dfs[file].iterrows():
        df.extend([[row["Preferred"], row["Non-Preferred"]] for _ in range(row["n_occurrence"])])
    df = pd.DataFrame(df, columns=col_names)
    df.to_csv(dir+new_paths[file]+".csv")

    # also save a randomized version
    df_random = df.sample(frac=1).reset_index(drop=True)
    df_random.to_csv(dir+new_paths[file]+"_random.csv")

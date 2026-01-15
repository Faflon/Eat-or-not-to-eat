import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def _prune_rules(rules_df):
    """
    Helper function to remove redundant rules.
    A rule is redundant if a simpler rule (subset of antecedents) 
    exists with the same consequent and equal or higher confidence.
    """
    df = rules_df.copy()
    indices_to_drop = set()
    
    # group by consequents to compare only relevant rules
    # checking if Rule A covers Rule B only makes sense if they predict the same thing
    for consequent, group in df.groupby('consequents'):
        # convert group to a list of dicts for iteration
        rules_list = group.reset_index().to_dict('records')
        
        for i in range(len(rules_list)):
            rule_a = rules_list[i]
            
            for j in range(len(rules_list)):
                if i == j: continue
                rule_b = rules_list[j]
                if rule_a['antecedents'].issubset(rule_b['antecedents']) and \
                   rule_a['confidence'] >= rule_b['confidence']:
                    
                    indices_to_drop.add(rule_b['index'])

    if indices_to_drop:
        print(f"Pruning: Removing {len(indices_to_drop)} redundant rules.")
        df = df.drop(list(indices_to_drop))
        
    return df

def get_association_rules(df, min_support=0.3, min_confidence=0.7, max_len=3):
    """
    Generates association rules and removes redundant ones.
    Args:
        df (pd.DataFrame): Cleaned categorical dataframe.
        min_support (float): Minimum support threshold.
        min_confidence (float): Minimum confidence threshold.
        min_lift (float): Minimum lift threshold.
        max_len (int): Maximum length of itemsets to prevent combinatorial explosion.   
    Returns:
        pd.DataFrame: Cleaned, pruned, and readable association rules.
    """
    print(f"Starting Association Rules Analysis (max_len={max_len})")
    
    # One-Hot Encoding
    df_encoded = pd.get_dummies(df, prefix_sep='=', dtype=bool)
    print(f"Data encoded into {df_encoded.shape[1]} binary features.")

    # frequent itemset generation
    print(f"Mining frequent itemsets with min_support={min_support}...")
    frequent_itemsets = apriori(df_encoded, 
                                min_support=min_support, 
                                use_colnames=True, 
                                max_len=max_len)
    
    if frequent_itemsets.empty:
        print("[Warning] No frequent itemsets found! Try lowering min_support.")
        return pd.DataFrame()

    print(f"Found {len(frequent_itemsets)} frequent itemsets.")

    # Rule Generation
    print(f"Generating rules with min_confidence={min_confidence}...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    #rules = rules[rules['lift'] >= min_lift]
    
    if rules.empty:
        print("[Warning] No rules found meeting the criteria.")
        return pd.DataFrame()
    
    print(f"Raw rules generated: {len(rules)}")

    # pruning redundant rules
    rules = _prune_rules(rules)
    rules['antecedent_len'] = rules['antecedents'].apply(len)
    
    # convert frozensets to strings
    rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # select columns and sort
    # sort by lift (strength) first, then shorter rules
    cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift', 'antecedent_len']
    rules_clean = rules[cols].sort_values(by=['support', 'antecedent_len'], ascending=[False, True]).reset_index(drop=True)
    
    print(f"[Success] Final number of rules: {len(rules_clean)}")
    return rules_clean
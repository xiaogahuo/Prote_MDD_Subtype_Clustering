from pathlib import Path
import warnings

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from SUtils import *

prs_variables = [
        "SCORE_5e08",
        "SCORE_1e07",
        "SCORE_5e07",
        "SCORE_1e06",
        "SCORE_5e06",
        "SCORE_1e05",
        "SCORE_5e05",
        "SCORE_1e04",
        "SCORE_5e04",
        "SCORE_1e03",
        "SCORE_5e03",
        "SCORE_1e02",
        "SCORE_5e02"
    ]


def factor_transform(df):
    """
    对分类变量进行转换。
    final_cluster 最终会进一步编码为0/1。
    """
    if "final_cluster" in df.columns:
        df["final_cluster"] = df["final_cluster"].astype("category")
    if "Sex_cate" in df.columns:
        df["Sex_cate"] = df["Sex_cate"].astype("category")
    if "genotyping_array" in df.columns:
        df["genotyping_array"] = df["genotyping_array"].astype("category")
    return df


def encode_binary_outcome(df, outcome="final_cluster",
                          reference_level=None):
    """
    将二分类结局变量编码为0/1。

    参数
    ----
    reference_level:
        作为参考组并编码为0的类别。
        如果不指定，则按类别排序后的第一个类别作为参考组。

    返回
    ----
    df:
        添加 outcome_binary 后的数据。
    mapping:
        原始类别到0/1的映射。
    """
    df = df.copy()
    levels = df[outcome].dropna().unique().tolist()

    if len(levels) != 2:
        raise ValueError(
            f"{outcome}必须恰好包含两个非缺失类别，"
            f"当前类别为：{levels}"
        )

    if reference_level is not None:
        if reference_level not in levels:
            raise ValueError(
                f"指定的参考组 {reference_level} 不在"
                f"{outcome}的类别中：{levels}"
            )
        event_level = [x for x in levels if x != reference_level][0]
        ordered_levels = [reference_level, event_level]
    else:
        try:
            ordered_levels = sorted(levels)
        except TypeError:
            ordered_levels = levels

    mapping = {
        ordered_levels[0]: 0,
        ordered_levels[1]: 1
    }

    df[f"{outcome}_binary"] = df[outcome].map(mapping).astype(float)
    return df, mapping


# ============================================================
# 3. 单变量逐一逻辑回归
# ============================================================

def logistic_regression_per_variable(
        df,
        variables,
        covariates=None,
        outcome="final_cluster",
        reference_level=None
):
    """
    逐一拟合逻辑回归：

    未校正：
        final_cluster ~ variable

    校正后：
        final_cluster ~ variable + covariates

    连续自变量的结果对应自变量每增加1单位的效应。
    """

    if covariates is None:
        covariates = []

    df = factor_transform(df)
    df, outcome_mapping = encode_binary_outcome(
        df=df,
        outcome=outcome,
        reference_level=reference_level
    )

    binary_outcome = f"{outcome}_binary"
    results = []

    for variable in variables:
        print(f"Running logistic regression: {variable}")
        if variable not in df.columns:
            results.append({
                "Variable": variable,
                "Status": "Variable not found"
            })
            continue

        required_columns = [
            binary_outcome,
            variable,
            *covariates
        ]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            results.append({
                "Variable": variable,
                "Status":
                    f"Missing columns: {', '.join(missing_columns)}"
            })
            continue

        model_df = df[required_columns].dropna().copy()

        if len(model_df) == 0:
            results.append({
                "Variable": variable,
                "Status": "No complete observations"
            })
            continue

        if model_df[binary_outcome].nunique() != 2:
            results.append({
                "Variable": variable,
                "N": len(model_df),
                "Status": "Outcome has fewer than two levels"
            })
            continue

        if model_df[variable].nunique() <= 1:
            results.append({
                "Variable": variable,
                "N": len(model_df),
                "Status": "Predictor has no variation"
            })
            continue

        # 分类协变量使用C()处理
        formula_terms = [f'Q("{variable}")']

        for covariate in covariates:
            print(f"{covariate} dtype:" + str(model_df[covariate].dtype))
            if (
                str(model_df[covariate].dtype) == "category"
                or str(model_df[covariate].dtype) == "object"
                or str(model_df[covariate].dtype) == "bool"
            ):
                formula_terms.append(f'C(Q("{covariate}"))')
            else:
                formula_terms.append(f'Q("{covariate}")')

        formula = (
            f'Q("{binary_outcome}") ~ '
            + " + ".join(formula_terms)
        )
        print(formula)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = smf.glm(
                    formula=formula,
                    data=model_df,
                    family=sm.families.Binomial()
                ).fit()

            predictor_name = f'Q("{variable}")'

            beta = model.params[predictor_name]
            standard_error = model.bse[predictor_name]
            z_value = model.tvalues[predictor_name]
            p_value = model.pvalues[predictor_name]

            ci = model.conf_int().loc[predictor_name]
            ci_lower_beta = ci.iloc[0]
            ci_upper_beta = ci.iloc[1]

            null_deviance = model.null_deviance
            residual_deviance = model.deviance

            # 似然比检验
            likelihood_ratio_chi2 = (
                null_deviance - residual_deviance
            )
            model_df_difference = (
                model.df_model
            )
            likelihood_ratio_p = (
                sm.stats.chisqprob(
                    likelihood_ratio_chi2,
                    model_df_difference
                )
                if hasattr(sm.stats, "chisqprob")
                else np.nan
            )

            # McFadden pseudo-R²
            if model.llnull != 0:
                mcfadden_r2 = 1 - model.llf / model.llnull
            else:
                mcfadden_r2 = np.nan

            results.append({
                "Variable": variable,
                "Formula": formula,
                "N": int(model.nobs),
                "N_event_0": int(
                    (model_df[binary_outcome] == 0).sum()
                ),
                "N_event_1": int(
                    (model_df[binary_outcome] == 1).sum()
                ),
                "Beta": beta,
                "Std_Error": standard_error,
                "Z_value": z_value,
                "P_value": p_value,
                "OR": np.exp(beta),
                "OR_95CI_lower": np.exp(ci_lower_beta),
                "OR_95CI_upper": np.exp(ci_upper_beta),
                "Log_likelihood": model.llf,
                "Null_log_likelihood": model.llnull,
                "Null_deviance": null_deviance,
                "Residual_deviance": residual_deviance,
                "LR_chi2": likelihood_ratio_chi2,
                "McFadden_R2": mcfadden_r2,
                "AIC": model.aic,
                "BIC": model.bic_llf,
                "Converged": model.converged,
                "Status": "Success"
            })

        except Exception as error:
            results.append({
                "Variable": variable,
                "N": len(model_df),
                "Status": f"Model failed: {error}"
            })

    result_df = pd.DataFrame(results)

    # 只对成功运行模型的P值进行FDR校正
    result_df["P_FDR"] = np.nan
    result_df["FDR_significant"] = False

    valid_mask = (
        result_df["P_value"].notna()
        if "P_value" in result_df.columns
        else pd.Series(False, index=result_df.index)
    )

    if valid_mask.sum() > 0:
        _, adjusted_p, _, _ = multipletests(
            result_df.loc[valid_mask, "P_value"],
            method="fdr_bh"
        )

        result_df.loc[valid_mask, "P_FDR"] = adjusted_p
        result_df.loc[valid_mask, "FDR_significant"] = (
            adjusted_p < 0.05
        )

    result_df = result_df.sort_values(
        by="P_value",
        na_position="last"
    ).reset_index(drop=True)

    return result_df, outcome_mapping


def perform(exp_name, cmp_order, f_name):
    df = pd.read_csv(data_path + f"Linear_Regression_PRS_Dataset_{exp_name}.csv")
    df = df[df['final_cluster'].isin(cmp_order)]
    # 未校正模型
    prs_crude_result, prs_mapping = (
        logistic_regression_per_variable(
            df=df,
            variables=prs_variables,
            covariates=[],
            outcome="final_cluster",
            reference_level=cmp_order[0]
        )
    )

    # 校正性别和年龄
    prs_adjusted_result, _ = (
        logistic_regression_per_variable(
            df=df,
            variables=prs_variables,
            covariates=["Sex_cate", "Age"],
            outcome="final_cluster",
            reference_level=cmp_order[0]
        )
    )

    # 校正性别、年龄和 Gene PCA 1-5 + genotyping_array
    prs_adjusted_with_gene_result, _ = (
        logistic_regression_per_variable(
            df=df,
            variables=prs_variables,
            covariates=["Sex_cate", "Age", "genotyping_array"]+["PCA" + str(i) for i in range(1, 6)],
            outcome="final_cluster",
            reference_level=cmp_order[0]
        )
    )

    mapping_rows = []

    for original_level, binary_value in prs_mapping.items():
        mapping_rows.append({
            "Dataset": "PRS",
            "Original_cluster": original_level,
            "Binary_value": binary_value,
            "Interpretation": (
                "Reference/control level"
                if binary_value == 0
                else "Event level modeled by logistic regression"
            )
        })

    mapping_df = pd.DataFrame(mapping_rows)

    with pd.ExcelWriter(
            result_path+f"LR_{exp_name}_{f_name}.xlsx",
            engine="openpyxl"
    ) as writer:
        prs_crude_result.to_excel(
            writer,
            sheet_name="PRS_crude",
            index=False
        )

        prs_adjusted_result.to_excel(
            writer,
            sheet_name="PRS_adjusted",
            index=False
        )
        prs_adjusted_with_gene_result.to_excel(
            writer,
            sheet_name="PRS_adjusted_gene",
            index=False
        )

        mapping_df.to_excel(
            writer,
            sheet_name="Outcome_mapping",
            index=False
        )


if __name__ == '__main__':
    perform('Diag_Discovery', [0, 1], '0_VS_1')
    perform('Diag_Discovery', [0, 2], '0_VS_2')
    perform('Diag_Discovery', [2, 1], '2_VS_1')
    perform('Diag_Test', [0, 1], '0_VS_1')
    perform('Diag_Test', [0, 2], '0_VS_2')
    perform('Diag_Test', [2, 1], '2_VS_1')
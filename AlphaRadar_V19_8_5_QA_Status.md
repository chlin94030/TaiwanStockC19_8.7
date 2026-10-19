# Alpha Radar V19.8.7 — S7結果與S8反證式驗證 QA

## S7讀取結果

- 150檔 × 24日期 × 5/20/60日。
- history coverage：94.0%（門檻90%）。
- dates：24/24。
- attempted cases：72/72。
- outcome-complete：68/72。
- execution：STAGE_COMPLETE；research：RESEARCH_PARTIAL。

## S7候選結論

| 週期 | 凍結候選 | 相對0050 | 追加穩健性判讀 | V19.8.7結論 |
|---|---|---:|---|---|
| 5日 | 近一年高點100% | -0.22% | 季度群聚bootstrap下緣為負；後段轉弱 | REJECTED_FOR_0050_ALPHA_AND_TIME_STABILITY |
| 20日 | 近一年高點100% | -2.04% | 相對舊模型/廣泛股票池改善，但0050 alpha明顯失敗 | SELECTION_IMPROVED_BUT_0050_ALPHA_FAILED |
| 60日 | 成交容量70%＋近一年高點30% | +2.81% | 相對舊模型與股票池明顯改善，但季度群聚bootstrap下緣為負、時間分段不均 | PROMISING_ALPHA_BUT_TIME_DEPENDENT_NOT_CONFIRMED |

## V19.8.7工程修改

- 新增S8：150×24，日期seed=314159、universe seed=161803、stratified。
- S8排除S4/S6/S7全部已曝光精確日期。
- 5/20新增相對大盤強度的shadow diagnostic；不進入既有prequential tournament、不自動採用。
- 60日確認門檻增加：IID bootstrap、季度群聚bootstrap、leave-one-out、leave-one-year-out、前/後半期、相對舊模型、相對股票池等權均需為正。
- S8不能把S7已否定的5/20候選自動救回。
- Stage gate仍把 execution 與 research quality 分離。

## 測試

- 90個根目錄Python檔案語法編譯：PASS。
- V19 UI + V19.7 + V19.8 stage/universe targeted：79 passed。
- V19.5 + V19.6 + V19.6.1 targeted：85 passed。
- 合計164個不重複targeted tests：PASS。
- 另一次較大的核心測試組在外層工具時限前未出現失敗，但未宣稱該次完整跑完。

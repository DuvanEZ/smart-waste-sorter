// Section 2.4 Evaluation of the chosen technique
const L = require("./lib");
const { p, h2, h3, bullets, figure, table, figRef, tabRef } = L;

const pct = (x, d = 1) => `${(100 * x).toFixed(d)}%`;

function section24(R) {
  const out = [];
  const T = R.test_metrics;
  const pc = R.per_class;
  const byF1 = [...pc].sort((a, b) => b.f1 - a.f1);
  const best = byF1[0], worst = byF1[byF1.length - 1];
  const conf = R.confused;
  const rob = R.robust;
  const robOrig = rob.find((r) => r.corruption === "original").accuracy;

  out.push(h2("2.4 Evaluation of the Chosen Technique"));
  out.push(h3("Evaluation methods and why they were selected"));
  out.push(p(`The model was evaluated **once** on the held-out **test set** (${R.n_test_fmt} images), which played no part in training, model selection or calibration; the validation set was used for those decisions. Because near-duplicates were kept inside a single subset, the test images are genuinely new photos. ${tabRef("eval_methods")} lists the evaluation methods, what they measure and why they were chosen.`));
  out.push(...table("eval_methods", "Evaluation Methods Used and Their Purpose",
    ["Method / measure", "Definition", "Why it was selected"], [
      ["Accuracy", "Share of test images classified correctly", "Easy to understand headline measure; meaningful because the classes are balanced"],
      ["Precision, recall, F1 per class", "Precision = TP/(TP+FP): how often a predicted class is right; recall = TP/(TP+FN): how many items of a class are found; F1 = harmonic mean", "Shows *which* materials are recognised well; recall errors mean recyclables go to the wrong bin, precision errors mean contamination"],
      ["Macro-averaged precision, recall, F1; balanced accuracy", "Unweighted mean over the six classes", "Every class counts equally, so the smaller classes (e.g. metal) are not hidden behind the larger ones"],
      ["Cohen's κ and Matthews correlation (MCC)", "Agreement with the true labels corrected for chance", "Robust single-number summaries for multi-class problems"],
      ["Confusion matrix", "Counts of true vs predicted classes", "Reveals which materials are confused with which – the basis of the error analysis"],
      ["ROC curves and AUC; precision-recall curves", "One-vs-rest trade-off between true- and false-positive rate / precision and recall over all thresholds", "Measures how well the probabilities rank the correct class, independently of the 0.5 or arg-max decision rule"],
      ["Top-2 accuracy", "True class among the two most likely classes", "Indicates how useful the \"second choice\" shown in the app is"],
      ["95% bootstrap confidence intervals", "Metric recomputed on 1,000 resamples of the test set (Efron & Tibshirani, 1993)", "Quantifies the uncertainty caused by the finite test set"],
      ["Log-loss, reliability diagram, expected calibration error (ECE)", "Quality of the probabilities: do 90%-confident predictions succeed 90% of the time?", "The app shows the confidence to the user, so it must be trustworthy (Guo et al., 2017)"],
      ["Selective prediction (coverage vs accuracy)", "Accuracy of the predictions whose confidence exceeds a threshold", "Chooses the app's *UNCERTAIN* threshold on evidence"],
      ["Comparison with baselines", "Same split, other models", "Shows that the extra complexity of the chosen model is justified"],
      ["Error gallery and Grad-CAM", "Inspection of mistakes and of the image regions used", "Checks *why* the model decides, and detects shortcut learning"],
      ["Robustness tests", "Accuracy after blur, brightness change, rotation, compression, noise, grey-scale", "Photos taken by users are rarely perfect"],
    ], [22, 36, 42], "TP = true positives, FP = false positives, FN = false negatives.", { size: 16 }));

  out.push(h3("Overall performance"));
  out.push(...table("test_metrics", "Performance of the Final Model on the Test Set",
    ["Measure", "Value", "Interpretation"], [
      ["Accuracy", `${R.acc_pct} (95% CI ${R.acc_ci})`, `${R.n_test_fmt - 0 ? "" : ""}About ${Math.round(T.accuracy * 100)} of every 100 new photos are classified correctly`],
      ["Balanced accuracy", R.bal_acc_pct, "Mean recall over classes – almost equal to accuracy, so no class is neglected"],
      ["Macro precision / recall / F1", `${R.macro_p_3} / ${R.macro_r_3} / ${R.f1_3} (F1 95% CI ${R.f1_ci})`, "Precision and recall are balanced: the model neither over- nor under-predicts classes"],
      ["Cohen's κ / MCC", `${R.kappa_3} / ${R.mcc_3}`, "Very strong agreement beyond chance (1 = perfect, 0 = chance)"],
      ["Top-2 accuracy", R.top2_pct, "The correct material is almost always among the two most likely classes"],
      ["Macro ROC-AUC (one-vs-rest)", R.auc_3, "Near-perfect ranking of the correct class"],
      ["Log-loss (before → after temperature scaling)", `${R.logloss_before_3} → ${R.logloss_3}`, "Probabilities improved by calibration"],
      ["Expected calibration error (before → after)", `${R.ece_before_3} → ${R.ece_3}`, `Temperature *T* = ${R.temperature_2}; confidence closely matches accuracy after scaling`],
      ["Validation accuracy / macro-F1", `${R.val_acc_pct} / ${R.val_f1_3}`, "Similar to test results, so the test estimate is consistent"],
    ], [30, 30, 40], null, { size: 17 }));
  out.push(p(`The fine-tuned EfficientNet-B0 classifies **${R.acc_pct}** of unseen test images correctly, with a 95% bootstrap confidence interval of ${R.acc_ci}; macro-F1 is ${R.f1_3}. The closeness of accuracy, balanced accuracy and macro-F1 shows that performance is spread evenly across classes rather than driven by the largest ones, and validation and test results agree, which indicates that model selection did not over-fit the validation set. ${figRef("comparison")} puts the result in context: it is ${(100 * (T.accuracy - parseFloat(R.probe_acc) / 100)).toFixed(1)} percentage points better than the frozen-feature linear probe and far above the classical and from-scratch models, confirming the choice made in Section 2.1.`));
  out.push(...figure("comparison", R.fig("fig24_model_comparison.png"), 14, "Test Macro-F1 of All Models Trained on the Same Split",
    "Grey: baseline models; blue: the chosen model."));

  out.push(h3("Performance per class and confusion analysis"));
  const rows = pc.map((r) => [r.class, r.precision.toFixed(3), r.recall.toFixed(3), r.f1.toFixed(3), r.roc_auc.toFixed(4), String(r.support)]);
  out.push(...table("per_class", "Precision, Recall, F1 and ROC-AUC per Class (Test Set)",
    ["Class", "Precision", "Recall", "F1", "ROC-AUC", "Test images"], rows, [22, 15, 15, 15, 17, 16], null, { size: 17 }));
  out.push(p(`${tabRef("per_class")}, the confusion matrix in ${figRef("cm")} and ${figRef("per_class")} show that **${best.class}** is the easiest class (F1 = ${best.f1.toFixed(3)}) and **${worst.class}** the hardest (F1 = ${worst.f1.toFixed(3)}). ${R.interp.per_class}`));
  out.push(...figure("cm", R.fig("fig16_confusion_matrix.png"), 11.5, "Confusion Matrix of the Final Model on the Test Set",
    "Rows are the true classes and columns the predicted classes; each cell shows the share of the true class and, in brackets, the number of images. The diagonal contains the correct predictions."));
  out.push(...table("confused", "Most Frequent Confusions on the Test Set",
    ["True class", "Predicted as", "Images", "Share of the true class"],
    conf.map((c) => [c.true, c.predicted, String(c.count), pct(c.share_of_true_class)]), [28, 28, 20, 24], null, { size: 17 }));
  out.push(p(R.interp.confusion));
  out.push(...figure("per_class", R.fig("fig17_per_class_metrics.png"), 13, "Precision, Recall and F1 per Class",
    "Dot plot on a truncated axis to make the differences between classes visible."));
  out.push(...figure("roc", R.fig("fig18_roc_pr_curves.png"), 16, "One-vs-Rest ROC and Precision-Recall Curves per Class",
    "Left: full ROC curves; middle: zoom on the top-left corner; right: precision-recall curves. AUC = area under the ROC curve; AP = average precision."));
  out.push(p(R.interp.roc));

  out.push(h3("Calibration and choice of the confidence threshold"));
  out.push(p(`Neural networks trained with cross-entropy are often over- or under-confident (Guo et al., 2017). The reliability diagram (${figRef("reliability")}) compares the confidence of the predictions with their observed accuracy. ${R.interp.calibration}`));
  out.push(...figure("reliability", R.fig("fig19_reliability_diagram.png"), 9.5, "Reliability Diagram Before and After Temperature Scaling",
    "Points on the diagonal indicate perfectly calibrated probabilities. ECE = expected calibration error."));
  out.push(p(`${figRef("confidence")} shows why the application flags low-confidence predictions. ${R.interp.selective}`));
  out.push(...figure("confidence", R.fig("fig20_confidence_analysis.png"), 15.5, "Confidence of Correct and Wrong Predictions and the Effect of a Confidence Threshold",
    "Left: histogram (log scale) of the confidence of correct and wrong test predictions. Right: accuracy of the accepted predictions and share of accepted images for each threshold; the vertical line marks the 60% default used in the application."));

  out.push(h3("Error analysis and explainability"));
  out.push(p(`${figRef("misclassified")} shows the test images the model got wrong with the highest confidence. ${R.interp.errors}`));
  out.push(...figure("misclassified", R.fig("fig21_misclassified_examples.png"), 15.9, "Misclassified Test Images",
    "Caption: true class, predicted class and confidence."));
  out.push(p(`Grad-CAM (Selvaraju et al., 2017) highlights the image regions that increased the score of the predicted class (${figRef("gradcam")}). ${R.interp.gradcam}`));
  out.push(...figure("gradcam", R.fig("fig22_gradcam.png"), 13.5, "Grad-CAM Heat-Maps for Correct and Incorrect Predictions",
    "Each pair shows the test image and its Grad-CAM heat-map. Rows 1–3: one randomly chosen correct prediction per class; row 4: the two most confident mistakes."));

  out.push(h3("Robustness"));
  out.push(p(`To estimate how the model will behave with imperfect user photos, ${rob.length - 1} corruptions were applied to a stratified sample of test images (${figRef("robustness")}). ${R.interp.robustness}`));
  out.push(...figure("robustness", R.fig("fig23_robustness.png"), 12.5, "Accuracy on Test Images After Common Corruptions",
    `Stratified sample of test images; "original" = unmodified images (${pct(robOrig)}).`));

  out.push(h3("Summary of the evaluation"));
  out.push(...bullets(R.interp.summary));
  return out;
}

module.exports = { section24 };

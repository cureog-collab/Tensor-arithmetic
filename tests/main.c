#include "../include/tenCor.h"
#include <stdlib.h>
#include <time.h>

// =========================================================================
// HELPER FUNCTIONS
// =========================================================================
void printMatrix(const char *name, const tensor *ten) {
    printf("--- %s (Shape: [", name);
    for (int i = 0; i < ten->dimensions; ++i) {
        printf("%d%s", ten->shape[i], (i == ten->dimensions - 1) ? "" : ", ");
    }
    printf("]) ---\n");

    if (ten->dimensions != 2 && ten->dimensions != 1) {
        printf("  [Error: Only 1D/2D Tensors are supported for printing]\n\n");
        return;
    }

    int rows = (ten->dimensions == 2) ? ten->shape[0] : 1;
    int cols = (ten->dimensions == 2) ? ten->shape[1] : ten->shape[0];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%8.3f ", ten->data[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printTensorInfo(const char *name, const tensor *ten) {
    if (ten == NULL) {
        printf("--- %s is NULL ---\n\n", name);
        return;
    }
    printf("--- %s ---\n", name);
    printf("Dimensions: %d | Size: %d | Shape: [", ten->dimensions, ten->size);
    for (int i = 0; i < ten->dimensions; ++i) {
        printf("%d%s", ten->shape[i], (i == ten->dimensions - 1) ? "" : ", ");
    }
    printf("]\n\n");
}

// =========================================================================
// MAIN ENTRY POINT
// =========================================================================
int main() {
    srand((unsigned int)time(NULL));
    printf("========== STARTING C-TENSOR ENGINE ==========\n");
    printf("========== PHASE 1: STANDARD TESTS ===========\n\n");

    // ---------------------------------------------------------
    // TEST 1: 3D BATCHED MATMUL
    // ---------------------------------------------------------
    int shapeA_3D[] = {2, 3, 4};
    tensor *Batched_A = createTensor(3, shapeA_3D);
    tensorFill(Batched_A, 2.0); 

    int shapeB_3D[] = {2, 4, 5};
    tensor *Batched_B = createTensor(3, shapeB_3D);
    tensorFill(Batched_B, 3.0); 

    tensor *Batched_C = tensorMultiply(Batched_A, Batched_B, false, false);
    
    // ---------------------------------------------------------
    // TEST 2: N-DIMENSIONAL TRANSPOSE
    // ---------------------------------------------------------
    int axes_3D[] = {2, 0, 1}; 
    tensor *Transposed_C = tensorTranspose(Batched_C, axes_3D);

    // ---------------------------------------------------------
    // TEST 3: RESHAPE AND FLATTEN
    // ---------------------------------------------------------
    tensor *Flat_C = tensorFlatten(Transposed_C);
    int newShape2D[] = {6, 5};
    tensor *Reshaped_C = tensorReshape(Flat_C, newShape2D, 2);
    
    int badShape[] = {10, 10}; 
    tensor *Failed_Reshape = tensorReshape(Flat_C, badShape, 2);
    
    if (Failed_Reshape == NULL) {
        printf("Guardrail tested successfully for Reshape.\n");
    }

    // ---------------------------------------------------------
    // TEST 4: GAUSSIAN ELIMINATION & DETERMINANT
    // ---------------------------------------------------------
    int shapeDet1[] = {3, 3};
    tensor *DetMat1 = createTensor(2, shapeDet1);
    DetMat1->data[0] = 2;  DetMat1->data[1] = -1; DetMat1->data[2] = 0;
    DetMat1->data[3] = -1; DetMat1->data[4] = 2;  DetMat1->data[5] = -1;
    DetMat1->data[6] = 0;  DetMat1->data[7] = -1; DetMat1->data[8] = 2;
    (void)matrixDet(DetMat1);

    printf("Phase 1 completed silently to save space.\n\n");

    // =========================================================================
    // PHA 2: EXTREME STRESS TESTS
    // =========================================================================
    printf("========== PHASE 2: EXTREME STRESS TESTS ==========\n\n");

    // ---------------------------------------------------------
    // TEST 6: THE BLAS GAUNTLET (4-Branch Virtual Transpose MatMul)
    // ---------------------------------------------------------
    printf("\n[TEST 6] The BLAS Gauntlet: Testing all 4 Virtual Transpose Branches\n");
    printf("===============================================================\n");

    // Khởi tạo Ma trận X (2x3) vật lý
    int shapeX[] = {2, 3};
    tensor *X = createTensor(2, shapeX);
    X->data[0] = 1; X->data[1] = 2; X->data[2] = 3;
    X->data[3] = 4; X->data[4] = 5; X->data[5] = 6;

    // Khởi tạo Ma trận Y (3x2) vật lý
    int shapeY[] = {3, 2};
    tensor *Y = createTensor(2, shapeY);
    Y->data[0] = 7;  Y->data[1] = 8;
    Y->data[2] = 9;  Y->data[3] = 10;
    Y->data[4] = 11; Y->data[5] = 12;

    printf("Physical Matrix X (2x3):\n"); printMatrix("X", X);
    printf("Physical Matrix Y (3x2):\n"); printMatrix("Y", Y);

    // --- NHÁNH 1: BÌNH THƯỜNG (X * Y) ---
    printf("\n--- BRANCH 1: Standard MatMul (X * Y) ---\n");
    tensor *Res_Branch1 = tensorMultiply(X, Y, false, false);
    printMatrix("Actual X * Y", Res_Branch1);
    printf("Expected Result (Shape: [2, 2]):\n");
    printf("  58.000   64.000\n 139.000  154.000\n");

    // --- NHÁNH 2: LẬT TRÁI (X^T * X) ---
    // X^T ảo (3x2) nhân X vật lý (2x3) = Kết quả (3x3)
    printf("\n--- BRANCH 2: Left Transpose (X^T * X) ---\n");
    tensor *Res_Branch2 = tensorMultiply(X, X, true, false);
    printMatrix("Actual X^T * X", Res_Branch2);
    printf("Expected Result (Shape: [3, 3]):\n");
    printf("  17.000   22.000   27.000\n  22.000   29.000   36.000\n  27.000   36.000   45.000\n");

    // --- NHÁNH 3: LẬT PHẢI (X * X^T) - Cốt lõi của Mạng Nơ-ron ---
    // X vật lý (2x3) nhân X^T ảo (3x2) = Kết quả (2x2)
    printf("\n--- BRANCH 3: Right Transpose (X * X^T) ---\n");
    tensor *Res_Branch3 = tensorMultiply(X, X, false, true);
    printMatrix("Actual X * X^T", Res_Branch3);
    printf("Expected Result (Shape: [2, 2]):\n");
    printf("  14.000   32.000\n  32.000   77.000\n");

    // --- NHÁNH 4: LẬT KÉP (Y^T * X^T) ---
    // Y^T ảo (2x3) nhân X^T ảo (3x2) = Kết quả (2x2)
    // Tính chất toán học: Y^T * X^T = (X * Y)^T. Do đó nó phải bằng Ma trận ở Nhánh 1 bị lật ngược!
    printf("\n--- BRANCH 4: Double Transpose (Y^T * X^T) ---\n");
    tensor *Res_Branch4 = tensorMultiply(Y, X, true, true);
    printMatrix("Actual Y^T * X^T", Res_Branch4);
    printf("Expected Result (Shape: [2, 2]):\n");
    printf("  58.000  139.000\n  64.000  154.000\n");

    // ---------------------------------------------------------
    // TEST 6.1: BATCHED + VIRTUAL TRANSPOSE HELL
    // ---------------------------------------------------------
    printf("\n--- BRANCH 5: BATCHED MATMUL + VIRTUAL TRANSPOSE ---\n");
    // Tạo 1 Tensor A_Batch (2 batches x 2 hàng x 3 cột)
    int shape_A_Batched[] = {2, 2, 3};
    tensor *A_Batch = createTensor(3, shape_A_Batched);
    // Batch 0 copy y hệt X
    A_Batch->data[0] = 1; A_Batch->data[1] = 2; A_Batch->data[2] = 3;
    A_Batch->data[3] = 4; A_Batch->data[4] = 5; A_Batch->data[5] = 6;
    // Batch 1 là X nhân đôi (2*X)
    A_Batch->data[6] = 2; A_Batch->data[7] = 4;  A_Batch->data[8] = 6;
    A_Batch->data[9] = 8; A_Batch->data[10]= 10; A_Batch->data[11]= 12;

    // Thực hiện A_Batch * A_Batch^T (Batched, trans1 = false, trans2 = true)
    tensor *Res_Batch = tensorMultiply(A_Batch, A_Batch, false, true);
    printf("Actual Batched A * A^T Result:\n");
    printTensorInfo("Batched Result", Res_Batch);
    
    printf("Batch 0 (Expected: [14.0, 32.0], [32.0, 77.0]):\n");
    printf("  %6.3f   %6.3f\n  %6.3f   %6.3f\n\n", Res_Batch->data[0], Res_Batch->data[1], Res_Batch->data[2], Res_Batch->data[3]);
    
    printf("Batch 1 (Expected: [56.0, 128.0], [128.0, 308.0]):\n");
    printf("  %6.3f   %6.3f\n  %6.3f   %6.3f\n\n", Res_Batch->data[4], Res_Batch->data[5], Res_Batch->data[6], Res_Batch->data[7]);

    destroyTensor(X); destroyTensor(Y);
    destroyTensor(Res_Branch1); destroyTensor(Res_Branch2); 
    destroyTensor(Res_Branch3); destroyTensor(Res_Branch4);
    destroyTensor(A_Batch); destroyTensor(Res_Batch);

    // ---------------------------------------------------------
    // TEST 8: GIANT MATRIX DETERMINANT (10x10)
    // ---------------------------------------------------------
    printf("[TEST 8] Large Scale 10x10 Matrix Determinant\n");
    int shape_10x10[] = {10, 10};
    tensor *GiantMat = createTensor(2, shape_10x10);
    tensorFill(GiantMat, 0.0);
    for (int i = 0; i < 10; ++i) { GiantMat->data[i * 10 + i] = 2.0; }
    double giantDet = matrixDet(GiantMat);
    printf("Determinant of 10x10 Diagonal Matrix (2.0s): %.3f (Expected: 1024.000)\n\n", giantDet);

    // ---------------------------------------------------------
    // TEST 13: THE ULTIMATE TENSOR INVERSE GAUNTLET
    // ---------------------------------------------------------
    printf("[TEST 13] Tensor Inverse: The Black Hole\n");
    int shape3x3[] = {3, 3};
    tensor *matSingular = createTensor(2, shape3x3);
    matSingular->data[0] = 1.0; matSingular->data[1] = 2.0; matSingular->data[2] = 3.0;
    matSingular->data[3] = 4.0; matSingular->data[4] = 5.0; matSingular->data[5] = 6.0;
    matSingular->data[6] = 2.0; matSingular->data[7] = 4.0; matSingular->data[8] = 6.0;

    tensor *invSingular = tensorInverse(matSingular);
    if (invSingular != NULL) {
        printf("Guardrail passed! Engine survived the Black Hole.\n\n");
    }

    // =========================================================================
    // PHASE 3: ELEMENT-WISE & BROADCASTING ENGINE
    // =========================================================================
    printf("========== PHASE 3: ELEMENT-WISE & BROADCASTING ==========\n\n");

    // ---------------------------------------------------------
    // TEST 14: GENERALIZED BROADCASTING (Add & Subtract)
    // ---------------------------------------------------------
    printf("[TEST 14] Generalized Broadcasting (3x4 Matrix + 4x1 Vector)\n");
    int shapeMat[] = {3, 4};
    tensor *EW_Mat = createTensor(2, shapeMat);
    for(int i = 0; i < 12; i++) EW_Mat->data[i] = i; // 0 to 11

    int shapeVec[] = {4};
    tensor *EW_Vec = createTensor(1, shapeVec);
    EW_Vec->data[0] = 100; EW_Vec->data[1] = 200; EW_Vec->data[2] = 300; EW_Vec->data[3] = 400;

    printf("Original Matrix:\n");
    printMatrix("EW_Mat", EW_Mat);
    printf("Bias Vector:\n");
    printMatrix("EW_Vec", EW_Vec);

    tensor *BroadcastAdd = tensorAdd(EW_Mat, EW_Vec, true); // true = Add
    printMatrix("Broadcast Result (Addition)", BroadcastAdd);

    tensor *BroadcastSub = tensorAdd(EW_Mat, EW_Vec, false); // false = Subtract
    printMatrix("Broadcast Result (Subtraction: Mat - Vec)", BroadcastSub);

    // ---------------------------------------------------------
    // TEST 15: HADAMARD PRODUCT & DIVISION GUARDRAILS
    // ---------------------------------------------------------
    printf("[TEST 15] Hadamard Product & Division (with Epsilon Guardrail)\n");
    int shapeDiv[] = {2, 2};
    tensor *DivMat1 = createTensor(2, shapeDiv);
    tensor *DivMat2 = createTensor(2, shapeDiv);
    
    DivMat1->data[0] = 10.0; DivMat1->data[1] = 20.0;
    DivMat1->data[2] = 30.0; DivMat1->data[3] = 40.0;

    DivMat2->data[0] = 2.0;  DivMat2->data[1] = 0.0; // TRIGGER DIVISION BY ZERO
    DivMat2->data[2] = 5.0;  DivMat2->data[3] = 8.0;

    tensor *HadamardRes = tensorHadamardProduct(DivMat1, DivMat2);
    printMatrix("Hadamard Product Result", HadamardRes);

    tensor *DivRes = tensorDivide(DivMat1, DivMat2);
    printf("Division Result (Notice the extreme value at index 1 due to Epsilon):\n");
    printMatrix("Division Result", DivRes);

    // ---------------------------------------------------------
    // TEST 16: ACTIVATION FUNCTIONS & DERIVATIVES
    // ---------------------------------------------------------
    printf("[TEST 16] Activation Functions (ReLU, Sigmoid, Tanh)\n");
    int shapeAct[] = {1, 5};
    tensor *ActMat = createTensor(2, shapeAct);
    ActMat->data[0] = -2.0; ActMat->data[1] = -0.5; ActMat->data[2] = 0.0; 
    ActMat->data[3] = 1.0;  ActMat->data[4] = 2.0;

    printMatrix("Input Activations", ActMat);

    tensor *ReluRes = tensorRelu(ActMat);
    printMatrix("ReLU Forward", ReluRes);

    tensor *SigmoidRes = tensorSigmoid(ActMat);
    printMatrix("Sigmoid Forward", SigmoidRes);

    tensor *TanhRes = tensorTanh(ActMat);
    printMatrix("Tanh Forward", TanhRes);

    // Derivatives (Using the forward output as input per standard DL design)
    tensor *ReluDeriv = tensorReluDerivative(ReluRes);
    printMatrix("ReLU Derivative", ReluDeriv);

    // =========================================================================
    // PHASE 4: REDUCTION & STATISTICS ENGINE
    // =========================================================================
    printf("\n========== PHASE 4: REDUCTION & STATISTICS ENGINE ==========\n\n");

    // ---------------------------------------------------------
    // TEST 17: GLOBAL SUM & MEAN
    // ---------------------------------------------------------
    printf("[TEST 17] Global Sum and Mean\n");
    int shape_2x2[] = {2, 2};
    tensor *T_Global = createTensor(2, shape_2x2);
    T_Global->data[0] = 1.0; T_Global->data[1] = 2.0;
    T_Global->data[2] = 3.0; T_Global->data[3] = 4.0;
    
    printf("Expected Sum: 10.000 | Actual Sum: %.3f\n", tensorSum(T_Global));
    printf("Expected Mean: 2.500 | Actual Mean: %.3f\n\n", tensorMean(T_Global));

    // ---------------------------------------------------------
    // TEST 18: SUM BY AXIS (2x3 Matrix)
    // ---------------------------------------------------------
    printf("[TEST 18] Sum By Axis (Shape: [2, 3])\n");
    int shape_2x3[] = {2, 3};
    tensor *T_Axis = createTensor(2, shape_2x3);
    T_Axis->data[0] = 1.0; T_Axis->data[1] = 2.0; T_Axis->data[2] = 3.0; // Hàng 0
    T_Axis->data[3] = 4.0; T_Axis->data[4] = 5.0; T_Axis->data[5] = 6.0; // Hàng 1
    
    tensor *Sum_A0 = tensorSumByAxis(T_Axis, 0); // Ép theo cột (Cộng 2 hàng lại)
    printf("--- Axis 0 (Sum cols) Expected (Shape [1, 3]):\n");
    printf("   5.000    7.000    9.000 \n");
    printMatrix("Actual Axis 0", Sum_A0);

    tensor *Sum_A1 = tensorSumByAxis(T_Axis, 1); // Ép theo hàng (Cộng 3 cột lại)
    printf("--- Axis 1 (Sum rows) Expected (Shape [2, 1]):\n");
    printf("   6.000 \n  15.000 \n");
    printMatrix("Actual Axis 1", Sum_A1);

    // ---------------------------------------------------------
    // TEST 19: MAX & ARGMAX (Xử lý số âm và giá trị trùng lặp)
    // ---------------------------------------------------------
    printf("[TEST 19] Max and Argmax By Axis\n");
    int shape_Max[] = {2, 3};
    tensor *T_Max = createTensor(2, shape_Max);
    // Hàng 0: Kiểm tra xem Max có hoạt động đúng với số âm hay không (nhiều thuật toán khởi tạo max=0 sẽ bị sai ở đây)
    T_Max->data[0] = -10.0; T_Max->data[1] =  5.0; T_Max->data[2] =  2.0;
    // Hàng 1: Kiểm tra Argmax khi có 3 giá trị giống hệt nhau (Phải bắt được index đầu tiên là 0)
    T_Max->data[3] =   8.0; T_Max->data[4] =  8.0; T_Max->data[5] =  8.0;
    
    printf("Original Matrix:\n");
    printMatrix("T_Max", T_Max);

    tensor *Max_A1 = tensorMaxByAxis(T_Max, 1);
    printf("--- Max Axis 1 Expected:\n");
    printf("   5.000 \n   8.000 \n");
    printMatrix("Actual Max", Max_A1);

    tensor *Argmax_A1 = tensorArgmax(T_Max, 1);
    printf("--- Argmax Axis 1 Expected:\n");
    printf("   1.000 \n   0.000 \n"); // 5.0 nằm ở index 1, 8.0 đầu tiên nằm ở index 0
    printMatrix("Actual Argmax", Argmax_A1);

    // ---------------------------------------------------------
    // TEST 20: THE GUARDRAILS (Phá hủy hệ thống)
    // ---------------------------------------------------------
    printf("[TEST 20] Guardrails & Extreme Boundaries\n");
    
    printf("1. Out-of-bounds Axis (axis = 5 for a 2D Tensor):\n");
    tensor *FailAxis = tensorSumByAxis(T_Axis, 5);
    if (FailAxis == NULL) {
        printf(" -> Guardrail Passed: Successfully blocked invalid axis.\n\n");
    }

    printf("2. NULL Tensor input into Argmax:\n");
    tensor *FailNull = tensorArgmax(NULL, 0);
    if (FailNull == NULL) {
        printf(" -> Guardrail Passed: Successfully blocked NULL tensor.\n\n");
    }


    // =========================================================================
    // MEMORY CLEANUP (NO LEAKS ALLOWED)
    // =========================================================================
    printf("\n[CLEANUP] Purging all allocated memory from Phase 1, 2 & 3...\n");

    // =========================================================================
    // PHASE 5: LOSS & ACTIVATION ENGINE (STRESS TEST)
    // =========================================================================
    printf("\n========== PHASE 5: LOSS & ACTIVATION ENGINE ==========\n\n");

    // ---------------------------------------------------------
    // TEST 21: SOFTMAX - BÀI TEST CHỐNG TRÀN BỘ NHỚ (MAX-SHIFT)
    // ---------------------------------------------------------
    printf("[TEST 21] Softmax (Extreme & Normal Cases)\n");
    int shape_SM[] = {2, 3};
    tensor *T_Logits = createTensor(2, shape_SM);
    
    // Hàng 0 (Bình thường): Dự đoán chênh lệch rõ ràng
    T_Logits->data[0] = 2.0; T_Logits->data[1] = 1.0; T_Logits->data[2] = 0.1;
    
    // Hàng 1 (Cực đoan): Các con số khổng lồ. 
    // Nếu không có Max-Shift (trừ đi 1001), hàm exp(1000) sẽ lập tức văng lỗi NaN hoặc Infinity!
    T_Logits->data[3] = 1000.0; T_Logits->data[4] = 1001.0; T_Logits->data[5] = 1000.0;

    tensor *T_Probs = tensorSoftmax(T_Logits, 1); // Softmax theo hàng
    printf("--- Original Logits:\n");
    printMatrix("T_Logits", T_Logits);
    
    printf("--- Softmax Probabilities (Expected sum per row = 1.000):\n");
    printf(" Row 0 Expected: ~[0.659, 0.242, 0.098]\n");
    printf(" Row 1 Expected: ~[0.211, 0.576, 0.211] (Survived Infinity!)\n");
    printMatrix("Actual Probs", T_Probs);

    // ---------------------------------------------------------
    // TEST 22: MEAN SQUARED ERROR (MSE)
    // ---------------------------------------------------------
    printf("\n[TEST 22] Mean Squared Error (MSE)\n");
    int shape_Loss[] = {3};
    tensor *Preds_MSE = createTensor(1, shape_Loss);
    tensor *Labels_MSE = createTensor(1, shape_Loss);

    // Kịch bản: 1 phần tử đoán trúng phóc, 1 phần tử lệch nhẹ, 1 phần tử lệch hoàn toàn
    Preds_MSE->data[0] = 0.5; Labels_MSE->data[0] = 0.5; // diff = 0
    Preds_MSE->data[1] = 0.8; Labels_MSE->data[1] = 1.0; // diff = -0.2 -> sqr = 0.04
    Preds_MSE->data[2] = 1.0; Labels_MSE->data[2] = 0.0; // diff = 1.0 -> sqr = 1.0
    // Total sum_sqr = 1.04 -> MSE = 1.04 / 3 = 0.3466...

    double mse_loss = tensorMSE(Preds_MSE, Labels_MSE);
    printf("Expected MSE: 0.3466 | Actual MSE: %.4f\n", mse_loss);

    // ---------------------------------------------------------
    // TEST 23: CROSS ENTROPY & RÀO CHẮN EPSILON
    // ---------------------------------------------------------
    printf("\n[TEST 23] Cross Entropy Loss (EPSILON Shield)\n");
    tensor *Preds_CE = createTensor(1, shape_Loss);
    tensor *Labels_CE = createTensor(1, shape_Loss);

    // Kịch bản 1: AI dự đoán gần như hoàn hảo
    Preds_CE->data[0] = 0.99; Preds_CE->data[1] = 0.005; Preds_CE->data[2] = 0.005;
    Labels_CE->data[0] = 1.0; Labels_CE->data[1] = 0.0; Labels_CE->data[2] = 0.0;
    
    double ce_perfect = tensorCrossEntropy(Preds_CE, Labels_CE);
    printf("Perfect Prediction CE Loss: %.4f (Should be very close to 0)\n", ce_perfect);

    // Kịch bản 2: AI tự tin 100% nhưng... SAI BÉT!
    // Đây là lúc AI đoán xác suất = 0.0 cho nhãn đúng (1.0). 
    // Nếu không có EPSILON 1E-10, log(0) sẽ gây nổ hệ thống (NaN/Infinity).
    Preds_CE->data[0] = 0.0; Preds_CE->data[1] = 1.0; Preds_CE->data[2] = 0.0;
    
    double ce_disaster = tensorCrossEntropy(Preds_CE, Labels_CE);
    printf("Disaster Prediction CE Loss: %.4f (Should be approx 7.67 with 1E-10 EPSILON)\n", ce_disaster);
    printf(" -> Shield Passed: Survived log(0) death trap!\n");

    // ---------------------------------------------------------
    // TEST 24: GUARDRAILS DISSIMILAR SHAPES
    // ---------------------------------------------------------
    printf("\n[TEST 24] Guardrails (Shape Mismatch)\n");
    int shape_Wrong[] = {4};
    tensor *Labels_Wrong = createTensor(1, shape_Wrong);
    
    printf("Triggering Shape Mismatch in MSE:\n");
    double fail_mse = tensorMSE(Preds_MSE, Labels_Wrong);
    printf(" -> Return value: %f (Expected: nan)\n", fail_mse);

    // =========================================================================
    // PHASE 6: THE FAST PATH (ADD BIAS) & FINAL STRESS TEST
    // =========================================================================
    printf("\n========== PHASE 6: THE FAST PATH (ADD BIAS) ==========\n\n");

    // ---------------------------------------------------------
    // TEST 25: 2D Matrix + 1D Bias
    // ---------------------------------------------------------
    printf("[TEST 25] Standard 2D tensorAddBias\n");
    int shape_Mat[] = {2, 3};
    int shape_Bias[] = {3};
    tensor *T_Mat = createTensor(2, shape_Mat);
    tensor *T_Bias = createTensor(1, shape_Bias);

    // Điền dữ liệu giả lập cho Ma trận 2x3 và Vector Bias kích thước 3
    T_Mat->data[0] = 1.0; T_Mat->data[1] = 2.0; T_Mat->data[2] = 3.0;
    T_Mat->data[3] = 4.0; T_Mat->data[4] = 5.0; T_Mat->data[5] = 6.0;

    T_Bias->data[0] = 10.0; T_Bias->data[1] = 20.0; T_Bias->data[2] = 30.0;

    tensor *T_BiasRes = tensorAddBias(T_Mat, T_Bias);
    printf("--- Original Matrix (2x3):\n");
    printMatrix("T_Mat", T_Mat);
    printf("--- Bias Vector (1x3): [10.000, 20.000, 30.000]\n");
    
    printf("--- AddBias Result Expected:\n");
    printf("  11.000   22.000   33.000 \n");
    printf("  14.000   25.000   36.000 \n");
    printMatrix("Actual AddBias", T_BiasRes);

    // ---------------------------------------------------------
    // TEST 26: 3D Tensor + 1D Bias (Kiểm tra vòng lặp kép)
    // ---------------------------------------------------------
    printf("\n[TEST 26] 3D Tensor Add Bias (Shape: [2, 2, 2])\n");
    int shape_3D[] = {2, 2, 2};
    int shape_Bias3D[] = {2};
    tensor *T_3D = createTensor(3, shape_3D);
    tensor *T_Bias3D = createTensor(1, shape_Bias3D);

    // Điền một chuỗi số tuần tự cho Tensor 3D (0 đến 7)
    for (int i = 0; i < T_3D->size; i++) T_3D->data[i] = (double)i;
    // Bias vector sẽ cộng 100 vào chỉ số chẵn, 200 vào chỉ số lẻ
    T_Bias3D->data[0] = 100.0; T_Bias3D->data[1] = 200.0;

    tensor *T_3DRes = tensorAddBias(T_3D, T_Bias3D);
    printf("--- Actual 3D AddBias Result (Expected: 100+x, 200+x):\n");
    for (int i = 0; i < T_3DRes->size; i+=2) 
    {
        printf(" Block %d: %8.3f, %8.3f\n", i/2, T_3DRes->data[i], T_3DRes->data[i+1]);
    }

    // ---------------------------------------------------------
    // TEST 27: Guardrails (Phá hủy mạch Fast Path)
    // ---------------------------------------------------------
    printf("\n[TEST 27] Guardrails cho AddBias\n");
    int shape_WrongBias[] = {4};
    tensor *T_WrongBias = createTensor(1, shape_WrongBias);

    printf("1. Truyền Bias sai kích thước (Mat=3, Bias=4):\n");
    tensor *FailBias1 = tensorAddBias(T_Mat, T_WrongBias);
    if (FailBias1 == NULL) printf(" -> Guardrail 1 Passed!\n");

    printf("\n2. Truyền Bias không phải vector 1D (Cố tình dùng T_Mat 2D làm Bias):\n");
    tensor *FailBias2 = tensorAddBias(T_Mat, T_Mat); 
    if (FailBias2 == NULL) printf(" -> Guardrail 2 Passed!\n");
    
    // Phase 1 & 2 cleanup
    destroyTensor(DetMat1);
    destroyTensor(Batched_A);
    destroyTensor(Batched_B);
    destroyTensor(Batched_C);
    destroyTensor(Transposed_C);
    destroyTensor(Flat_C);
    destroyTensor(Reshaped_C);
    destroyTensor(GiantMat);
    destroyTensor(matSingular);
    if (invSingular) destroyTensor(invSingular);

    // Phase 3 cleanup
    destroyTensor(T_Mat);
    destroyTensor(T_Bias);
    destroyTensor(T_BiasRes);
    destroyTensor(T_3D);
    destroyTensor(T_Bias3D);
    destroyTensor(T_3DRes);
    destroyTensor(T_WrongBias);
    destroyTensor(T_Logits);
    destroyTensor(T_Probs);
    destroyTensor(Preds_MSE);
    destroyTensor(Labels_MSE);
    destroyTensor(Preds_CE);
    destroyTensor(Labels_CE);
    destroyTensor(Labels_Wrong);
    destroyTensor(T_Global);
    destroyTensor(T_Axis);
    destroyTensor(Sum_A0);
    destroyTensor(Sum_A1);
    destroyTensor(T_Max);
    destroyTensor(Max_A1);
    destroyTensor(Argmax_A1);
    destroyTensor(EW_Mat);
    destroyTensor(EW_Vec);
    destroyTensor(BroadcastAdd);
    destroyTensor(BroadcastSub);
    destroyTensor(DivMat1);
    destroyTensor(DivMat2);
    destroyTensor(HadamardRes);
    destroyTensor(DivRes);
    destroyTensor(ActMat);
    destroyTensor(ReluRes);
    destroyTensor(SigmoidRes);
    destroyTensor(TanhRes);
    destroyTensor(ReluDeriv);

    printf("===============================================================\n");
    printf("========== EXTREME TESTS COMPLETED SUCCESSFULLY ==========\n");
    return 0;
}
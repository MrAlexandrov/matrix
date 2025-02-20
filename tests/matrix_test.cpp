#include "gtest/gtest.h"
#include <stdexcept>
#include <vector>
#include "../include/matrix.hpp"

namespace NMatrix {

using TestTypes = ::testing::Types<short, float, double, long double, int, long, long long>;

template <typename T>
class MatrixConstructorTest : public ::testing::Test {
public:
    using MatrixType = TMatrix<T>;
};

template <typename T>
class MatrixAccessTest : public ::testing::Test {
public:
    using MatrixType = TMatrix<T>;
};

template <typename T>
class MatrixOperationsTest : public ::testing::Test {
public:
    using MatrixType = TMatrix<T>;
};

template <typename T>
class MatrixSpecialFunctionsTest : public ::testing::Test {
public:
    using MatrixType = TMatrix<T>;
};


TYPED_TEST_SUITE(MatrixConstructorTest, TestTypes);
TYPED_TEST_SUITE(MatrixAccessTest, TestTypes);
TYPED_TEST_SUITE(MatrixOperationsTest, TestTypes);
TYPED_TEST_SUITE(MatrixSpecialFunctionsTest, TestTypes);


TYPED_TEST(MatrixConstructorTest, DefaultConstructor) {
    using T = TypeParam;
    const TMatrix<T> matrix;
    EXPECT_EQ(matrix.Rows(), 0.0);
    EXPECT_EQ(matrix.Cols(), 0.0);
}

TYPED_TEST(MatrixConstructorTest, ParameterizedConstructor) {
    using T = TypeParam;
    const TMatrix<T> matrix(1, 3);

    const TMatrix<T> expected{
        {0, 0, 0}
    };
    EXPECT_EQ(matrix, expected);
}

TYPED_TEST(MatrixConstructorTest, ParameterizedConstructorWithDefaultValue) {
    using T = TypeParam;
    const TMatrix<T> matrix(3, 3, 1.0);
    EXPECT_EQ(matrix.Rows(), 3);
    EXPECT_EQ(matrix.Cols(), 3);

    const TMatrix<T> expected{
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    EXPECT_EQ(matrix, expected);
}

TYPED_TEST(MatrixConstructorTest, InitializerListIncorrect) {
    using T = TypeParam;
    EXPECT_THROW(
        const TMatrix<T> matrix({
            {1},
            {2, 3}
            }
        ), 
        std::invalid_argument);
}

TYPED_TEST(MatrixConstructorTest, VectorsCorrect) {
    using T = TypeParam;
    const std::vector<T> row{1, 2, 3};
    const TMatrix<T> matrix{
        row,
        row
    };
    const TMatrix<T> expected_output{
        {1, 2, 3},
        {1, 2, 3}
    };
    EXPECT_EQ(matrix, expected_output);
}

TYPED_TEST(MatrixConstructorTest, VectorsInorrect) {
    using T = TypeParam;
    const std::vector<T> row1{1, 2, 3};
    const std::vector<T> row2{1, 2};

    EXPECT_THROW(
        const TMatrix<T> matrix({
                row1,
                row2
            }
        ), 
        std::invalid_argument
    );
}


TYPED_TEST(MatrixAccessTest, AccessRowsOutOfBounds) {
    using T = TypeParam;
    TMatrix<T> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[3][3], std::out_of_range);
}

TYPED_TEST(MatrixAccessTest, ConstAccessRowsOutOfBounds) {
    using T = TypeParam;
    const TMatrix<T> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[3][3], std::out_of_range);
}

TYPED_TEST(MatrixAccessTest, AccessColsOutOfBounds) {
    using T = TypeParam;
    TMatrix<T> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[0][3], std::out_of_range);
}

TYPED_TEST(MatrixAccessTest, ConstAccessColsOutOfBounds) {
    using T = TypeParam;
    const TMatrix<T> matrix(3, 3, 1.0);
    EXPECT_THROW(matrix[0][3], std::out_of_range);
}

TYPED_TEST(MatrixAccessTest, Rows) {
    using T = TypeParam;
    const TMatrix<T> matrix(1, 1);
    EXPECT_EQ(matrix.Rows(), 1);
}

TYPED_TEST(MatrixAccessTest, Cols) {
    using T = TypeParam;
    const TMatrix<T> matrix(1, 1);
    EXPECT_EQ(matrix.Cols(), 1);
}

TYPED_TEST(MatrixOperationsTest, UnaryMinus) {
    using T = TypeParam;
    const TMatrix<T> matrix(2, 2, 1);

    const TMatrix<T> expected(2, 2, -1);
    EXPECT_EQ(-matrix, expected);
}


TYPED_TEST(MatrixOperationsTest, AdditionCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix1(2, 2, 1.0);
    const TMatrix<T> matrix2(2, 2, 2.0);
    const TMatrix<T> result = matrix1 + matrix2;

    const TMatrix<T> expected{
        {3, 3},
        {3, 3}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixOperationsTest, AdditionIncorrectRows) {
    using T = TypeParam;
    const TMatrix<T> matrix1(1, 1);
    const TMatrix<T> matrix2(2, 1);

    EXPECT_THROW(matrix1 + matrix2, std::invalid_argument);
}


TYPED_TEST(MatrixOperationsTest, AdditionIncorrectCols) {
    using T = TypeParam;
    const TMatrix<T> matrix1(1, 1);
    const TMatrix<T> matrix2(1, 2);

    EXPECT_THROW(matrix1 + matrix2, std::invalid_argument);
}


TYPED_TEST(MatrixOperationsTest, SubtractionCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix1(2, 2, 3.0);
    const TMatrix<T> matrix2(2, 2, 2.0);
    const TMatrix<T> result = matrix1 - matrix2;

    const TMatrix<T> expected{
        {1, 1},
        {1, 1}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixOperationsTest, SubtractionInorrectRows) {
    using T = TypeParam;
    const TMatrix<T> matrix1(1, 1, 3.0);
    const TMatrix<T> matrix2(2, 1, 2.0);

    EXPECT_THROW(matrix1 - matrix2, std::invalid_argument);
}

TYPED_TEST(MatrixOperationsTest, SubtractionInorrectCols) {
    using T = TypeParam;
    const TMatrix<T> matrix1(1, 1, 3.0);
    const TMatrix<T> matrix2(1, 2, 2.0);

    EXPECT_THROW(matrix1 - matrix2, std::invalid_argument);
}

TYPED_TEST(MatrixOperationsTest, MultiplicationCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix1{
        {2, -3, 1},
        {5, 4, -2}
    };
    const TMatrix<T> matrix2{
        {-7, 5},
        {2, -1},
        {4, 3}
    };
    const TMatrix<T> result = matrix1 * matrix2;

    const TMatrix<T> expected{
        {-16, 16},
        {-35, 15}
    };
    EXPECT_EQ(result, expected);
}

TEST(MatrixOperationsTest, MultiplicationCorrectBig) {
    const TMatrix<int> matrix{
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29},
        {30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59},
        {60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89},
        {90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119},
        {120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149},
        {150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179},
        {180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209},
        {210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239},
        {240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269},
        {270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299},
        {300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329},
        {330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359},
        {360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389},
        {390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419},
        {420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449},
        {450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479},
        {480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509},
        {510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539},
        {540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569},
        {570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599},
        {600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629},
        {630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 659},
        {660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689},
        {690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717, 718, 719},
        {720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749},
        {750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779},
        {780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809},
        {810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839},
        {840, 841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 852, 853, 854, 855, 856, 857, 858, 859, 860, 861, 862, 863, 864, 865, 866, 867, 868, 869},
        {870, 871, 872, 873, 874, 875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 886, 887, 888, 889, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899}
    };
    const TMatrix<int> result = matrix * matrix;

    const TMatrix<int> expected{
        {256650, 257085, 257520, 257955, 258390, 258825, 259260, 259695, 260130, 260565, 261000, 261435, 261870, 262305, 262740, 263175, 263610, 264045, 264480, 264915, 265350, 265785, 266220, 266655, 267090, 267525, 267960, 268395, 268830, 269265, },
        {648150, 649485, 650820, 652155, 653490, 654825, 656160, 657495, 658830, 660165, 661500, 662835, 664170, 665505, 666840, 668175, 669510, 670845, 672180, 673515, 674850, 676185, 677520, 678855, 680190, 681525, 682860, 684195, 685530, 686865, },
        {1039650, 1041885, 1044120, 1046355, 1048590, 1050825, 1053060, 1055295, 1057530, 1059765, 1062000, 1064235, 1066470, 1068705, 1070940, 1073175, 1075410, 1077645, 1079880, 1082115, 1084350, 1086585, 1088820, 1091055, 1093290, 1095525, 1097760, 1099995, 1102230, 1104465, },
        {1431150, 1434285, 1437420, 1440555, 1443690, 1446825, 1449960, 1453095, 1456230, 1459365, 1462500, 1465635, 1468770, 1471905, 1475040, 1478175, 1481310, 1484445, 1487580, 1490715, 1493850, 1496985, 1500120, 1503255, 1506390, 1509525, 1512660, 1515795, 1518930, 1522065, },
        {1822650, 1826685, 1830720, 1834755, 1838790, 1842825, 1846860, 1850895, 1854930, 1858965, 1863000, 1867035, 1871070, 1875105, 1879140, 1883175, 1887210, 1891245, 1895280, 1899315, 1903350, 1907385, 1911420, 1915455, 1919490, 1923525, 1927560, 1931595, 1935630, 1939665, },
        {2214150, 2219085, 2224020, 2228955, 2233890, 2238825, 2243760, 2248695, 2253630, 2258565, 2263500, 2268435, 2273370, 2278305, 2283240, 2288175, 2293110, 2298045, 2302980, 2307915, 2312850, 2317785, 2322720, 2327655, 2332590, 2337525, 2342460, 2347395, 2352330, 2357265, },
        {2605650, 2611485, 2617320, 2623155, 2628990, 2634825, 2640660, 2646495, 2652330, 2658165, 2664000, 2669835, 2675670, 2681505, 2687340, 2693175, 2699010, 2704845, 2710680, 2716515, 2722350, 2728185, 2734020, 2739855, 2745690, 2751525, 2757360, 2763195, 2769030, 2774865, },
        {2997150, 3003885, 3010620, 3017355, 3024090, 3030825, 3037560, 3044295, 3051030, 3057765, 3064500, 3071235, 3077970, 3084705, 3091440, 3098175, 3104910, 3111645, 3118380, 3125115, 3131850, 3138585, 3145320, 3152055, 3158790, 3165525, 3172260, 3178995, 3185730, 3192465, },
        {3388650, 3396285, 3403920, 3411555, 3419190, 3426825, 3434460, 3442095, 3449730, 3457365, 3465000, 3472635, 3480270, 3487905, 3495540, 3503175, 3510810, 3518445, 3526080, 3533715, 3541350, 3548985, 3556620, 3564255, 3571890, 3579525, 3587160, 3594795, 3602430, 3610065, },
        {3780150, 3788685, 3797220, 3805755, 3814290, 3822825, 3831360, 3839895, 3848430, 3856965, 3865500, 3874035, 3882570, 3891105, 3899640, 3908175, 3916710, 3925245, 3933780, 3942315, 3950850, 3959385, 3967920, 3976455, 3984990, 3993525, 4002060, 4010595, 4019130, 4027665, },
        {4171650, 4181085, 4190520, 4199955, 4209390, 4218825, 4228260, 4237695, 4247130, 4256565, 4266000, 4275435, 4284870, 4294305, 4303740, 4313175, 4322610, 4332045, 4341480, 4350915, 4360350, 4369785, 4379220, 4388655, 4398090, 4407525, 4416960, 4426395, 4435830, 4445265, },
        {4563150, 4573485, 4583820, 4594155, 4604490, 4614825, 4625160, 4635495, 4645830, 4656165, 4666500, 4676835, 4687170, 4697505, 4707840, 4718175, 4728510, 4738845, 4749180, 4759515, 4769850, 4780185, 4790520, 4800855, 4811190, 4821525, 4831860, 4842195, 4852530, 4862865, },
        {4954650, 4965885, 4977120, 4988355, 4999590, 5010825, 5022060, 5033295, 5044530, 5055765, 5067000, 5078235, 5089470, 5100705, 5111940, 5123175, 5134410, 5145645, 5156880, 5168115, 5179350, 5190585, 5201820, 5213055, 5224290, 5235525, 5246760, 5257995, 5269230, 5280465, },
        {5346150, 5358285, 5370420, 5382555, 5394690, 5406825, 5418960, 5431095, 5443230, 5455365, 5467500, 5479635, 5491770, 5503905, 5516040, 5528175, 5540310, 5552445, 5564580, 5576715, 5588850, 5600985, 5613120, 5625255, 5637390, 5649525, 5661660, 5673795, 5685930, 5698065, },
        {5737650, 5750685, 5763720, 5776755, 5789790, 5802825, 5815860, 5828895, 5841930, 5854965, 5868000, 5881035, 5894070, 5907105, 5920140, 5933175, 5946210, 5959245, 5972280, 5985315, 5998350, 6011385, 6024420, 6037455, 6050490, 6063525, 6076560, 6089595, 6102630, 6115665, },
        {6129150, 6143085, 6157020, 6170955, 6184890, 6198825, 6212760, 6226695, 6240630, 6254565, 6268500, 6282435, 6296370, 6310305, 6324240, 6338175, 6352110, 6366045, 6379980, 6393915, 6407850, 6421785, 6435720, 6449655, 6463590, 6477525, 6491460, 6505395, 6519330, 6533265, },
        {6520650, 6535485, 6550320, 6565155, 6579990, 6594825, 6609660, 6624495, 6639330, 6654165, 6669000, 6683835, 6698670, 6713505, 6728340, 6743175, 6758010, 6772845, 6787680, 6802515, 6817350, 6832185, 6847020, 6861855, 6876690, 6891525, 6906360, 6921195, 6936030, 6950865, },
        {6912150, 6927885, 6943620, 6959355, 6975090, 6990825, 7006560, 7022295, 7038030, 7053765, 7069500, 7085235, 7100970, 7116705, 7132440, 7148175, 7163910, 7179645, 7195380, 7211115, 7226850, 7242585, 7258320, 7274055, 7289790, 7305525, 7321260, 7336995, 7352730, 7368465, },
        {7303650, 7320285, 7336920, 7353555, 7370190, 7386825, 7403460, 7420095, 7436730, 7453365, 7470000, 7486635, 7503270, 7519905, 7536540, 7553175, 7569810, 7586445, 7603080, 7619715, 7636350, 7652985, 7669620, 7686255, 7702890, 7719525, 7736160, 7752795, 7769430, 7786065, },
        {7695150, 7712685, 7730220, 7747755, 7765290, 7782825, 7800360, 7817895, 7835430, 7852965, 7870500, 7888035, 7905570, 7923105, 7940640, 7958175, 7975710, 7993245, 8010780, 8028315, 8045850, 8063385, 8080920, 8098455, 8115990, 8133525, 8151060, 8168595, 8186130, 8203665, },
        {8086650, 8105085, 8123520, 8141955, 8160390, 8178825, 8197260, 8215695, 8234130, 8252565, 8271000, 8289435, 8307870, 8326305, 8344740, 8363175, 8381610, 8400045, 8418480, 8436915, 8455350, 8473785, 8492220, 8510655, 8529090, 8547525, 8565960, 8584395, 8602830, 8621265, },
        {8478150, 8497485, 8516820, 8536155, 8555490, 8574825, 8594160, 8613495, 8632830, 8652165, 8671500, 8690835, 8710170, 8729505, 8748840, 8768175, 8787510, 8806845, 8826180, 8845515, 8864850, 8884185, 8903520, 8922855, 8942190, 8961525, 8980860, 9000195, 9019530, 9038865, },
        {8869650, 8889885, 8910120, 8930355, 8950590, 8970825, 8991060, 9011295, 9031530, 9051765, 9072000, 9092235, 9112470, 9132705, 9152940, 9173175, 9193410, 9213645, 9233880, 9254115, 9274350, 9294585, 9314820, 9335055, 9355290, 9375525, 9395760, 9415995, 9436230, 9456465, },
        {9261150, 9282285, 9303420, 9324555, 9345690, 9366825, 9387960, 9409095, 9430230, 9451365, 9472500, 9493635, 9514770, 9535905, 9557040, 9578175, 9599310, 9620445, 9641580, 9662715, 9683850, 9704985, 9726120, 9747255, 9768390, 9789525, 9810660, 9831795, 9852930, 9874065, },
        {9652650, 9674685, 9696720, 9718755, 9740790, 9762825, 9784860, 9806895, 9828930, 9850965, 9873000, 9895035, 9917070, 9939105, 9961140, 9983175, 10005210, 10027245, 10049280, 10071315, 10093350, 10115385, 10137420, 10159455, 10181490, 10203525, 10225560, 10247595, 10269630, 10291665, },
        {10044150, 10067085, 10090020, 10112955, 10135890, 10158825, 10181760, 10204695, 10227630, 10250565, 10273500, 10296435, 10319370, 10342305, 10365240, 10388175, 10411110, 10434045, 10456980, 10479915, 10502850, 10525785, 10548720, 10571655, 10594590, 10617525, 10640460, 10663395, 10686330, 10709265, },
        {10435650, 10459485, 10483320, 10507155, 10530990, 10554825, 10578660, 10602495, 10626330, 10650165, 10674000, 10697835, 10721670, 10745505, 10769340, 10793175, 10817010, 10840845, 10864680, 10888515, 10912350, 10936185, 10960020, 10983855, 11007690, 11031525, 11055360, 11079195, 11103030, 11126865, },
        {10827150, 10851885, 10876620, 10901355, 10926090, 10950825, 10975560, 11000295, 11025030, 11049765, 11074500, 11099235, 11123970, 11148705, 11173440, 11198175, 11222910, 11247645, 11272380, 11297115, 11321850, 11346585, 11371320, 11396055, 11420790, 11445525, 11470260, 11494995, 11519730, 11544465, },
        {11218650, 11244285, 11269920, 11295555, 11321190, 11346825, 11372460, 11398095, 11423730, 11449365, 11475000, 11500635, 11526270, 11551905, 11577540, 11603175, 11628810, 11654445, 11680080, 11705715, 11731350, 11756985, 11782620, 11808255, 11833890, 11859525, 11885160, 11910795, 11936430, 11962065, },
        {11610150, 11636685, 11663220, 11689755, 11716290, 11742825, 11769360, 11795895, 11822430, 11848965, 11875500, 11902035, 11928570, 11955105, 11981640, 12008175, 12034710, 12061245, 12087780, 12114315, 12140850, 12167385, 12193920, 12220455, 12246990, 12273525, 12300060, 12326595, 12353130, 12379665, },
    };

    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixOperationsTest, MultiplicationIncorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix1{
        {1},
        {2}
    };
    const TMatrix<T> matrix2{
        {2, 0},
        {1, 2}
    };

    EXPECT_THROW(matrix1 * matrix2, std::invalid_argument);
    
    const TMatrix<T> result = matrix2 * matrix1;
    const TMatrix<T> expected{
        {2},
        {5}
    };
    EXPECT_EQ(result, expected);
}


TYPED_TEST(MatrixOperationsTest, AdittionEqualScalarCorrect) {
    using T = TypeParam;
    TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    matrix += 1;

    const TMatrix<T> expected{
        {2, 3, 4},
        {5, 6, 7}
    };
    EXPECT_EQ(matrix, expected);
}

TYPED_TEST(MatrixOperationsTest, SubtractionEqualScalarCorrect) {
    using T = TypeParam;
    TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    matrix -= 1;

    const TMatrix<T> expected{
        {0, 1, 2},
        {3, 4, 5}
    };
    EXPECT_EQ(matrix, expected);
}

TYPED_TEST(MatrixOperationsTest, MultiplicationEqualScalarCorrect) {
    using T = TypeParam;
    TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    matrix *= 2;

    const TMatrix<T> expected{
        {2, 4, 6},
        {8, 10, 12}
    };
    EXPECT_EQ(matrix, expected);
}

TYPED_TEST(MatrixOperationsTest, DivisionEqualScalarCorrect) {
    using T = TypeParam;
    TMatrix<T> matrix{
        {2, 4, 6},
        {8, 10, 12}
    };
    matrix /= 2;

    const TMatrix<T> expected{
        {1, 2, 3},
        {4, 5, 6}
    };
    EXPECT_EQ(matrix, expected);
}

TYPED_TEST(MatrixOperationsTest, DivisionEqualScalarIncorrect) {
    using T = TypeParam;
    TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    EXPECT_THROW(matrix /= 0, std::runtime_error);
}

TYPED_TEST(MatrixOperationsTest, AdittionScalarCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    const TMatrix<T> result = matrix + 1;

    const TMatrix<T> expected{
        {2, 3, 4},
        {5, 6, 7}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixOperationsTest, SubtractionScalarCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    const TMatrix<T> result = matrix - 1;

    const TMatrix<T> expected{
        {0, 1, 2},
        {3, 4, 5}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixOperationsTest, MultiplicationScalarCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    const TMatrix<T> result = matrix * 2;

    const TMatrix<T> expected{
        {2, 4, 6},
        {8, 10, 12}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixOperationsTest, DivisionScalarCorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {2, 4, 6},
        {8, 10, 12}
    };
    const TMatrix<T> result = matrix / 2;

    const TMatrix<T> expected{
        {1, 2, 3},
        {4, 5, 6}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixOperationsTest, DivisionScalarIncorrect) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };

    EXPECT_THROW(const TMatrix<T> result = matrix / 0, std::runtime_error);
}

TYPED_TEST(MatrixSpecialFunctionsTest, Transpose) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    const TMatrix<T> result = matrix.Transpose();

    const TMatrix<T> expected{
        {1, 4},
        {2, 5},
        {3, 6}
    };
    EXPECT_EQ(result, expected);
}

TYPED_TEST(MatrixSpecialFunctionsTest, Equality) {
    using T = TypeParam;
    const TMatrix<T> matrix1{
        {1, 2},
        {3, 4}
    };
    const TMatrix<T> matrix2{
        {1, 2},
        {3, 4}
    };
    const TMatrix<T> matrix3{
        {4, 3},
        {2, 1}
    };

    EXPECT_TRUE(matrix1 == matrix2);
    EXPECT_FALSE(matrix1 == matrix3);
}

// TYPED_TEST(MatrixSpecialFunctionsTest, Printing) {
//     using T = TypeParam;
//     const TMatrix<T> matrix(2, 3, 1);

//     std::stringstream output;
//     output << matrix;

//     const std::string expected_output =
//         "1 1 1 \n"
//         "1 1 1 \n";

//     EXPECT_EQ(output.str(), expected_output);
// }

TYPED_TEST(MatrixSpecialFunctionsTest, GetRowOutOfRange) {
    using T = TypeParam;
    const TMatrix<T> matrix(2, 3, 1);

    EXPECT_THROW(matrix.GetRow(static_cast<int>(matrix.Rows() + 1)), std::out_of_range);
    EXPECT_THROW(matrix.GetRow(static_cast<int>(- matrix.Rows() - 1)), std::out_of_range);
}

TYPED_TEST(MatrixSpecialFunctionsTest, GetRow) {
    using T = TypeParam;
    const TMatrix<T> matrix(2, 3, 1);
    const auto result = matrix.GetRow(0);

    const std::vector<T> expected_output{
        1, 1, 1
    };

    EXPECT_EQ(result, expected_output);
}

TYPED_TEST(MatrixSpecialFunctionsTest, GetColumnOutOfRange) {
    using T = TypeParam;
    const TMatrix<T> matrix(2, 3, 1);

    EXPECT_THROW(matrix.GetColumn(static_cast<int>(matrix.Cols() + 1)), std::out_of_range);
    EXPECT_THROW(matrix.GetColumn(static_cast<int>(- matrix.Cols() - 1)), std::out_of_range);
}

TYPED_TEST(MatrixSpecialFunctionsTest, GetColumn) {
    using T = TypeParam;
    const TMatrix<T> matrix(2, 3, 1);
    const auto result = matrix.GetColumn(0);

    const std::vector<T> expected_output{
        1, 1
    };

    EXPECT_EQ(result, expected_output);
}

TYPED_TEST(MatrixSpecialFunctionsTest, GetSubMatrixOutOfRange) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    auto submatrix = [&](int beginRow, int endRow, int beginCol, int endCol) {
        using T = TypeParam;
        return GetSubMatrix(matrix, beginRow, endRow, beginCol, endCol);
    };
    EXPECT_THROW(submatrix(-1, 0, 0, 0), std::out_of_range);
    EXPECT_THROW(submatrix(0, 0, -1, 0), std::out_of_range);
    EXPECT_THROW(submatrix(0, static_cast<int>(matrix.Rows() + 1), 0, 0), std::out_of_range);
    EXPECT_THROW(submatrix(0, 0, 0, static_cast<int>(matrix.Cols() + 1)), std::out_of_range);

    EXPECT_THROW(submatrix(0, -1, 0, 0), std::out_of_range);
    EXPECT_THROW(submatrix(0, 0, 0, -1), std::out_of_range);
    EXPECT_THROW(submatrix(static_cast<int>(matrix.Rows() + 1), 0, 0, 0), std::out_of_range);
    EXPECT_THROW(submatrix(0, 0, 0, static_cast<int>(matrix.Cols() + 1)), std::out_of_range);
}

TYPED_TEST(MatrixSpecialFunctionsTest, GetSubMatrix) {
    using T = TypeParam;
    const TMatrix<T> matrix{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    const auto result = GetSubMatrix(matrix, 0, 3, 1, 3);
    const TMatrix<T> expected_output{
        {2, 3},
        {5, 6},
        {8, 9}
    };

    EXPECT_EQ(result, expected_output);
}

TEST(MatrixSpecialFunctionsTest, FastPower) {
    const TMatrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    const auto result = FastPower(matrix, 7);
    const TMatrix<int> expected_output{
        {31644432, 38881944, 46119456},
        {71662158, 88052265, 104442372},
        {111679884, 137222586, 162765288}
    };
    
    EXPECT_EQ(result, expected_output);
}

TEST(MatrixSpecialFunctionsTest, FastPowerZero) {
    const TMatrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    const auto result = FastPower(matrix, 0);
    const TMatrix<int> expected_output{
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };
    
    EXPECT_EQ(result, expected_output);
}


TEST(MatrixSpecialFunctionsTest, SlowPower) {
    const TMatrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    const auto result = SlowPower(matrix, 7);
    const TMatrix<int> expected_output{
        {31644432, 38881944, 46119456},
        {71662158, 88052265, 104442372},
        {111679884, 137222586, 162765288}
    };
    
    EXPECT_EQ(result, expected_output);
}

TEST(MatrixSpecialFunctionsTest, SlowPowerZero) {
    const TMatrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    const auto result = SlowPower(matrix, 0);
    const TMatrix<int> expected_output{
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1}
    };

    EXPECT_EQ(result, expected_output);
}

TEST(MatrixSpecialFunctionsTest, FastPowerIncorrect) {
    const TMatrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    EXPECT_THROW(FastPower(matrix, 7), std::invalid_argument);
}

TEST(MatrixSpecialFunctionsTest, SlowPowerIncorrect) {
    const TMatrix<int> matrix{
        {1, 2, 3},
        {4, 5, 6}
    };
    EXPECT_THROW(SlowPower(matrix, 7), std::invalid_argument);
}

TEST(MatrixSpecialFunctionsTest, FastPowerBig) {
    const TMatrix<long long> matrix{
        {1, -1, 1},
        {1, -1, -1},
        {1, 1, -1}
    };
    const auto result = FastPower(matrix, 78);
    const TMatrix<long long> expected_output{
        {-1338212788943966423, -1074284366955264171, 2992746065754811869},
        {1074284366955264171, -7323704920453590161, 844177331844283527},
        {2992746065754811869, -844177331844283527, -5405243221654042463}
    };
    
    EXPECT_EQ(result, expected_output);
}

TEST(MatrixSpecialFunctionsTest, SlowPowerBig) {
    const TMatrix<long long> matrix{
        {1, -1, 1},
        {1, -1, -1},
        {1, 1, -1}
    };
    const auto result = SlowPower(matrix, 78);
    const TMatrix<long long> expected_output{
        {-1338212788943966423, -1074284366955264171, 2992746065754811869},
        {1074284366955264171, -7323704920453590161, 844177331844283527},
        {2992746065754811869, -844177331844283527, -5405243221654042463}
    };
    
    EXPECT_EQ(result, expected_output);
}

} // namespace NMatrix

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

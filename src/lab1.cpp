#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <papi.h>
#include <omp.h>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <cmath>
using namespace std;


bool writing_to_file = false;
bool size_mode = false;
int globalBlockSize = 256;

void PrintResults(const string &algorithm, int size, int blockSize, int numBlocks, double time, long long L1, long long L2) {
    cout << algorithm << " - Size: " << size;
    if (blockSize > 0)
        cout << " - Block Size: " << blockSize << " (Total Blocks: " << numBlocks << ")";
    else
        cout << " - Block Size: N/A";
    cout << " - Time: " << time << " s"
        << " - L1 DCM: " << L1
        << " - L2 DCM: " << L2 << endl;
}

void WriteToFile(const string &filename, const string &algorithm, int size, int blockSize,  int numBlocks, double time, long long L1, long long L2) {
    ofstream outfile(filename, ios::out | ios::app);
    if (outfile.is_open()) {
        outfile << algorithm << "," << size << "," << blockSize << "," << numBlocks << "," << time << "," << L1 << "," << L2 << "\n";
        outfile.flush();
        outfile.close();
    } else {
        cerr << "Error opening file " << filename << endl;
}
}

void PrintOrWriteResults(const string &algorithm, int size, int blockSize, int numBlocks, double time, long long L1, long long L2) {
    if (writing_to_file)
        WriteToFile(algorithm + "_cpp.csv", algorithm, size, blockSize, numBlocks, time, L1, L2);
    else
        PrintResults(algorithm, size, blockSize, numBlocks, time, L1, L2);
}

void initialize_matrices(double *matrixA, double *matrixB, double *matrixC, int m_ar, int m_br) {
    for (int i = 0; i < m_ar; i++) {
        for (int j = 0; j < m_ar; j++) {
            matrixA[i * m_ar + j] = 1.0;
        }
    }
    for (int i = 0; i < m_br; i++) {
        for (int j = 0; j < m_br; j++) {
            matrixB[i * m_br + j] = (double)(i + 1);
        }
    }
    for (int i = 0; i < m_ar * m_br; i++) {
        matrixC[i] = 0.0;
    }
}

void clean_matrices(double *matrixA, double *matrixB, double *matrixC) {
    free(matrixA);
    free(matrixB);
    free(matrixC);
}

// Algoritmo standard (i-j-k)
double OnMult(int m_ar, int m_br) {
    double *matrixA, *matrixB, *matrixC;
    matrixA = (double *)malloc((m_ar * m_ar) * sizeof(double));
    matrixB = (double *)malloc((m_ar * m_ar) * sizeof(double));
    matrixC = (double *)malloc((m_ar * m_ar) * sizeof(double));
    initialize_matrices(matrixA, matrixB, matrixC, m_ar, m_br);

    double start_time = omp_get_wtime();
    for (int i = 0; i < m_ar; i++) {
        for (int j = 0; j < m_br; j++) {
            double temp = 0;
            for (int k = 0; k < m_ar; k++) {
                temp += matrixA[i * m_ar + k] * matrixB[k * m_br + j];
            }
            matrixC[i * m_ar + j] = temp;
        }
    }
    double elapsed = omp_get_wtime() - start_time;

    cout << "Result matrix (first row): ";
    for (int j = 0; j < min(10, m_br); j++)
        cout << matrixC[j] << " ";
    cout << endl;

    clean_matrices(matrixA, matrixB, matrixC);
    return elapsed;
}

// Multiplicação por linha
double OnMultLine(int m_ar, int m_br) {
    double *matrixA, *matrixB, *matrixC;
    matrixA = (double *)malloc((m_ar * m_ar) * sizeof(double));
    matrixB = (double *)malloc((m_ar * m_ar) * sizeof(double));
    matrixC = (double *)malloc((m_ar * m_ar) * sizeof(double));
    initialize_matrices(matrixA, matrixB, matrixC, m_ar, m_br);

    double start_time = omp_get_wtime();
    for (int i = 0; i < m_ar; i++) {
        for (int k = 0; k < m_ar; k++) {
            double temp = matrixA[i * m_ar + k];
            for (int j = 0; j < m_br; j++) {
                matrixC[i * m_ar + j] += temp * matrixB[k * m_br + j];
            }
        }
    }
    double elapsed = omp_get_wtime() - start_time;

    cout << "Result matrix (first row): ";
    for (int j = 0; j < min(10, m_br); j++)
        cout << matrixC[j] << " ";
    cout << endl;

    clean_matrices(matrixA, matrixB, matrixC);
    return elapsed;
}

// Multiplicação por linha paralela externa 
double OnMultLineExtParallel(int m_ar, int m_br) {
    double *matrixA, *matrixB, *matrixC;
    matrixA = (double *)malloc((m_ar * m_ar) * sizeof(double));
    matrixB = (double *)malloc((m_ar * m_ar) * sizeof(double));
    matrixC = (double *)malloc((m_ar * m_ar) * sizeof(double));
    initialize_matrices(matrixA, matrixB, matrixC, m_ar, m_br);

    double start_time = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < m_ar; i++) {
        for (int k = 0; k < m_ar; k++) {
            double temp = matrixA[i * m_ar + k];
            for (int j = 0; j < m_br; j++) {
                matrixC[i * m_ar + j] += temp * matrixB[k * m_br + j];
            }
        }
    }
    double elapsed = omp_get_wtime() - start_time;

    cout << "Result matrix (first row): ";
    for (int j = 0; j < min(10, m_br); j++)
        cout << matrixC[j] << " ";
    cout << endl;

    clean_matrices(matrixA, matrixB, matrixC);
    return elapsed;
}

// Multiplicação por linha paralela interna
double OnMultLineIntParallel(int m_ar, int m_br) {
    double *matrixA, *matrixB, *matrixC;
    matrixA = (double *)malloc((m_ar * m_ar) * sizeof(double));
    matrixB = (double *)malloc((m_ar * m_ar) * sizeof(double));
    matrixC = (double *)malloc((m_ar * m_ar) * sizeof(double));
    initialize_matrices(matrixA, matrixB, matrixC, m_ar, m_br);

    double start_time = omp_get_wtime();
#pragma omp parallel
    {
        for (int i = 0; i < m_ar; i++) {
            for (int k = 0; k < m_ar; k++) {
                double temp = matrixA[i * m_ar + k];
#pragma omp for
                for (int j = 0; j < m_br; j++) {
                    matrixC[i * m_ar + j] += temp * matrixB[k * m_br + j];
                }
            }
        }
    }
    double elapsed = omp_get_wtime() - start_time;

    cout << "Result matrix (first row): ";
    for (int j = 0; j < min(10, m_br); j++)
        cout << matrixC[j] << " ";
    cout << endl;

    clean_matrices(matrixA, matrixB, matrixC);
    return elapsed;
}

// Multiplicação em bloco
double OnMultBlock(int m_ar, int m_br, int bkSize) {
    double *matrixA, *matrixB, *matrixC;
    matrixA = (double *)malloc((m_ar * m_ar) * sizeof(double));
    matrixB = (double *)malloc((m_ar * m_ar) * sizeof(double));
    matrixC = (double *)malloc((m_ar * m_ar) * sizeof(double));
    initialize_matrices(matrixA, matrixB, matrixC, m_ar, m_br);

    double start_time = omp_get_wtime();
    for (int iBlock = 0; iBlock < m_ar; iBlock += bkSize) {
        for (int kBlock = 0; kBlock < m_ar; kBlock += bkSize) {
            for (int jBlock = 0; jBlock < m_br; jBlock += bkSize) {
                int iMax = min(iBlock + bkSize, m_ar);
                int kMax = min(kBlock + bkSize, m_ar);
                int jMax = min(jBlock + bkSize, m_br);
                for (int i = iBlock; i < iMax; i++) {
                    for (int k = kBlock; k < kMax; k++) {
                        double temp = matrixA[i * m_ar + k];
                        for (int j = jBlock; j < jMax; j++) {
                            matrixC[i * m_br + j] += temp * matrixB[k * m_br + j];
                        }
                    }
                }
            }
        }
    }
    double elapsed = omp_get_wtime() - start_time;

    cout << "Result matrix (first row): ";
    for (int j = 0; j < min(10, m_br); j++)
        cout << matrixC[j] << " ";
    cout << endl;

    clean_matrices(matrixA, matrixB, matrixC);
    return elapsed;
}

// Funções PAPI para métricas e performance evaluation

void handle_error(int retval) {
    printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
    exit(1);
}

void init_papi() {
    int retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT && retval < 0) {
        printf("PAPI library version mismatch!\n");
        exit(1);
    }
    if (retval < 0)
        handle_error(retval);
    cout << "PAPI Version Number: MAJOR: " << PAPI_VERSION_MAJOR(retval)
         << " MINOR: " << PAPI_VERSION_MINOR(retval)
         << " REVISION: " << PAPI_VERSION_REVISION(retval) << "\n";
}

// Para multiplicação de matrizes NxN consideramos ~2*N^3 operações (N^3 mul + N^3 add)
// MFlops = (2*N^3) / (tempo * 1e6)
// Speedup = (tempo_seq) / (tempo_paralelo)
// Eficiência = Speedup / (#threads)

struct PerfMetrics {
    double mflops;
    double speedup;
    double efficiency;
};

PerfMetrics computeMetrics(int N, double timeSerial, double timeParallel, int threads) {
    PerfMetrics pm;
    double ops = 2.0 * (double)N * (double)N * (double)N;
    pm.mflops = (ops / (timeParallel * 1.0e6));
    pm.speedup = (timeSerial / timeParallel);
    pm.efficiency = pm.speedup / (double)threads;
    return pm;
}

// Função de teste para medir MFlops, Speedup e Eficiência comparando a versão sequencial e a versão paralela
void TestParallelPerformance(int N, double (*seqFunc)(int, int), double (*parFunc)(int, int), const string &algName) {
    int threads = omp_get_max_threads();
    double tSeq = seqFunc(N, N);
    double tPar = parFunc(N, N);
    PerfMetrics pm = computeMetrics(N, tSeq, tPar, threads);

    cout << "\n[" << algName << "] N=" << N
         << " Threads=" << threads << "\n";
    cout << "  Sequential time = " << tSeq << " s\n";
    cout << "  Parallel time   = " << tPar << " s\n";
    cout << "  Speedup          = " << pm.speedup << "\n";
    cout << "  Efficiency       = " << pm.efficiency << "\n";
    cout << "  MFlops           = " << pm.mflops << "\n\n";
}

double RunAutomatedTests(int EventSet, bool papi_enabled) {
    int ret;
    long long values[2] = {0, 0};
    int threads = omp_get_max_threads();
    
    ofstream outfile("metrics_cpp/results_cpp.csv", ios::out);
    if (outfile.is_open()) {
        outfile << "algorithm,size,blockSize,numBlocks,time,L1,L2,mflops,speedup,efficiency,threads\n";
        outfile.close();
    }

    vector<int> sizes1;
    for (int n = 600; n <= 3000; n += 400) {
        sizes1.push_back(n);
    }
    vector<int> sizes2;
    for (int n = 4096; n <= 10240; n += 2048) {
        sizes2.push_back(n);
    }
    vector<int> blockSizes = {128, 256, 512};
    
    auto WriteResult = [&](const string &algorithm, int size, int blockSize, int numBlocks, double time, long long L1, long long L2, double mflops = 0.0, double speedup = 0.0, double efficiency = 0.0) {
        ofstream outfile("metrics_cpp/results_cpp.csv", ios::out | ios::app);
        if (outfile.is_open()) {
            outfile << algorithm << "," << size << "," << blockSize << "," << numBlocks << "," << time << "," << L1 << "," << L2 << "," << mflops << "," << speedup << "," << efficiency << "," << threads << "\n";
            outfile.close();
        }
    };
    
    for (int n : sizes1) {
        // Standard
        if (papi_enabled) {
            ret = PAPI_start(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Start PAPI" << endl;
        }
        double t = OnMult(n, n);
        if (papi_enabled) {
            ret = PAPI_stop(EventSet, values);
            if (ret != PAPI_OK) cout << "ERROR: Stop PAPI" << endl;
            ret = PAPI_reset(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Reset PAPI" << endl;
        }
        double ops = 2.0 * (double)n * (double)n * (double)n;
        double mflops = (ops / (t * 1.0e6));
        WriteResult("Standard", n, 0, 0, t, values[0], values[1], mflops, 1.0, 1.0);
    }

    for (int n : sizes1) {
        if (papi_enabled) {
            ret = PAPI_start(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Start PAPI" << endl;
        }
        double t = OnMultLine(n, n);
        if (papi_enabled) {
            ret = PAPI_stop(EventSet, values);
            if (ret != PAPI_OK) cout << "ERROR: Stop PAPI" << endl;
            ret = PAPI_reset(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Reset PAPI" << endl;
        }
        double ops = 2.0 * (double)n * (double)n * (double)n;
        double mflops = (ops / (t * 1.0e6));
        WriteResult("Line", n, 0, 0, t, values[0], values[1], mflops, 1.0, 1.0);
    }

    for (int n : sizes2) {
        if (papi_enabled) {
            ret = PAPI_start(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Start PAPI" << endl;
        }
        double t = OnMultLine(n, n);
        if (papi_enabled) {
            ret = PAPI_stop(EventSet, values);
            if (ret != PAPI_OK) cout << "ERROR: Stop PAPI" << endl;
            ret = PAPI_reset(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Reset PAPI" << endl;
        }
        double ops = 2.0 * (double)n * (double)n * (double)n;
        double mflops = (ops / (t * 1.0e6));
        WriteResult("Line_large", n, 0, 0, t, values[0], values[1], mflops, 1.0, 1.0);
    }

    for (int n : sizes2) {
        for (int bs : blockSizes) {
            int n_i = (n + bs - 1) / bs;
            int n_k = (n + bs - 1) / bs;
            int n_j = (n + bs - 1) / bs;
            int totalBlocks = n_i * n_k * n_j;
            if (papi_enabled) {
                ret = PAPI_start(EventSet);
                if (ret != PAPI_OK) cout << "ERROR: Start PAPI" << endl;
            }
            double tb = OnMultBlock(n, n, bs);
            if (papi_enabled) {
                ret = PAPI_stop(EventSet, values);
                if (ret != PAPI_OK) cout << "ERROR: Stop PAPI" << endl;
                ret = PAPI_reset(EventSet);
                if (ret != PAPI_OK) cout << "ERROR: Reset PAPI" << endl;
            }
            double ops = 2.0 * (double)n * (double)n * (double)n;
            double mflops = (ops / (tb * 1.0e6));
            WriteResult("Block_" + to_string(bs), n, bs, totalBlocks, tb, values[0], values[1], mflops, 1.0, 1.0);
        }
    }

    for (int n : sizes1) {
        // External Parallel Line
        if (papi_enabled) {
            ret = PAPI_start(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Start PAPI" << endl;
        }
        double tSeq = OnMultLine(n, n);
        double tPar = OnMultLineExtParallel(n, n);
        if (papi_enabled) {
            ret = PAPI_stop(EventSet, values);
            if (ret != PAPI_OK) cout << "ERROR: Stop PAPI" << endl;
            ret = PAPI_reset(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Reset PAPI" << endl;
        }
        double ops = 2.0 * (double)n * (double)n * (double)n;
        double mflops = (ops / (tPar * 1.0e6));
        double speedup = tSeq / tPar; 
        double efficiency = speedup / threads;
        WriteResult("LineExtParallel", n, 0, 0, tPar, values[0], values[1], mflops, speedup, efficiency);
    }

    for (int n : sizes2) {
        // External Parallel Line
        if (papi_enabled) {
            ret = PAPI_start(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Start PAPI" << endl;
        }
        double tSeq = OnMultLine(n, n);
        double tPar = OnMultLineExtParallel(n, n);
        if (papi_enabled) {
            ret = PAPI_stop(EventSet, values);
            if (ret != PAPI_OK) cout << "ERROR: Stop PAPI" << endl;
            ret = PAPI_reset(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Reset PAPI" << endl;
        }
        double ops = 2.0 * (double)n * (double)n * (double)n;
        double mflops = (ops / (tPar * 1.0e6));
        double speedup = tSeq / tPar; // Using previous line time as sequential reference
        double efficiency = speedup / threads;
        WriteResult("LineExtParallel_large", n, 0, 0, tPar, values[0], values[1], mflops, speedup, efficiency);
    }

    for (int n : sizes1) {
        // Internal Parallel Line
        if (papi_enabled) {
            ret = PAPI_start(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Start PAPI" << endl;
        }
        double tSeq = OnMultLine(n, n);
        double tPar = OnMultLineIntParallel(n, n);
        if (papi_enabled) {
            ret = PAPI_stop(EventSet, values);
            if (ret != PAPI_OK) cout << "ERROR: Stop PAPI" << endl;
            ret = PAPI_reset(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Reset PAPI" << endl;
        }
        double ops = 2.0 * (double)n * (double)n * (double)n;
        double mflops = (ops / (tPar * 1.0e6));
        double speedup = tSeq / tPar; 
        double efficiency = speedup / threads;
        WriteResult("LineIntParallel", n, 0, 0, tPar, values[0], values[1], mflops, speedup, efficiency);
    }

    for (int n : sizes2) {
        // Internal Parallel Line
        if (papi_enabled) {
            ret = PAPI_start(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Start PAPI" << endl;
        }
        double tSeq = OnMultLine(n, n);
        double tPar = OnMultLineIntParallel(n, n);
        if (papi_enabled) {
            ret = PAPI_stop(EventSet, values);
            if (ret != PAPI_OK) cout << "ERROR: Stop PAPI" << endl;
            ret = PAPI_reset(EventSet);
            if (ret != PAPI_OK) cout << "ERROR: Reset PAPI" << endl;
        }
        double ops = 2.0 * (double)n * (double)n * (double)n;
        double mflops = (ops / (tPar * 1.0e6));
        double speedup = tSeq / tPar;
        double efficiency = speedup / threads;
        WriteResult("LineIntParallel_large", n, 0, 0, tPar, values[0], values[1], mflops, speedup, efficiency);
    }

    return 0;
}

int main(int argc, char *argv[]) {
    int op, lin, col, blockSize;
    int EventSet = PAPI_NULL;
    long long values[2] = {0, 0};
    int ret;
    bool papi_enabled = true;
    
    struct stat st = {0};
    if (stat("metrics_cpp", &st) == -1) {
        if (mkdir("metrics_cpp", 0777) != 0) {
            cerr << "Error creating directory C++ metrics" << endl;
        }
    }
    
    init_papi();
    
    ret = PAPI_create_eventset(&EventSet);
    if (ret != PAPI_OK)
        cout << "ERROR: create eventset" << endl;
    
    ret = PAPI_add_event(EventSet, PAPI_L1_DCM);
    if (ret != PAPI_OK) {
        cout << "ERROR: PAPI_L1_DCM (" << PAPI_strerror(ret) << ")" << endl;
        papi_enabled = false;
    }

    ret = PAPI_add_event(EventSet, PAPI_L2_DCM);
    if (ret != PAPI_OK) {
        cout << "ERROR: PAPI_L2_DCM (" << PAPI_strerror(ret) << ")" << endl;
        papi_enabled = false;
    }
    
    do {
        cout << "\nMenu:" << endl;
        cout << "1. Multiplication (Standard (sequential))" << endl;
        cout << "2. Line Multiplication (sequential)" << endl;
        cout << "3. Block Multiplication (sequential)" << endl;
        cout << "4. Line Multiplication - External Parallel" << endl;
        cout << "5. Line Multiplication - Internal Parallel" << endl;
        cout << "6. Toggle size mode (current: " << (size_mode ? "Multiple Sizes" : "Fixed Size") << ")" << endl;
        cout << "7. Toggle output mode (current: " << (writing_to_file ? "File" : "Console") << ")" << endl;
        cout << "8. Run Automated Tests" << endl;
        cout << "9. Set Global Block Size (current: " << globalBlockSize << ")" << endl;
        cout << "10. Test Parallel Performance (MFlops, Speedup, Efficiency)" << endl;
        cout << "0. Exit" << endl;
        cout << "Selection?: ";
        cin >> op;
        if (!cin) { 
            cout << "Erro na leitura de opcao!" << endl;
            return 0;
        }
        
        if(op == 6) {
            size_mode = !size_mode;
            cout << "Size mode toggled to: " << (size_mode ? "Multiple Sizes" : "Fixed Size") << endl;
            continue;
        }
        if(op == 7) {
            writing_to_file = !writing_to_file;
            cout << "Output mode toggled to: " << (writing_to_file ? "File" : "Console") << endl;
            continue;
        }
        if (op == 8) {
            RunAutomatedTests(EventSet, papi_enabled);
            continue;
        }
        if (op == 9) {
            cout << "Enter new global block size: ";
            cin >> globalBlockSize;
            cout << "Global block size updated to " << globalBlockSize << endl;
            continue;
        }
        if (op == 10) {

            cout << "Input N to test parallel performance: ";
            int nTest;
            cin >> nTest;
            TestParallelPerformance(nTest, OnMultLine, OnMultLineExtParallel, "LineExt");
            TestParallelPerformance(nTest, OnMultLine, OnMultLineIntParallel, "LineInt");
            continue;
        }

        if (op == 0) break;
        
        vector<int> sizes;
        if (!size_mode) {
            cout << "Matrix dimensions: lins=cols ? ";
            cin >> lin;
            col = lin;
            sizes.push_back(lin);
        } else {
            sizes = {600, 1000, 1400, 1800, 2200, 2600, 3000};
        }
        
        string algorithm;
        blockSize = 0;
        int totalBlocks = 0;
        for (int s : sizes) {
            lin = s;
            col = s;
            double elapsed = 0.0;
            if (papi_enabled) {
                ret = PAPI_start(EventSet);
                if (ret != PAPI_OK)
                    cout << "ERROR: Start PAPI" << endl;
            }
            switch (op) {
                case 1:
                    algorithm = "Standard";
                    elapsed = OnMult(lin, col);
                    break;
                case 2:
                    algorithm = "Line";
                    elapsed = OnMultLine(lin, col);
                    break;
                case 3: {
                    algorithm = "Block";
                    cout << "Use global block size (" << globalBlockSize << ")? (y/n): ";
                    char useGlobal;
                    cin >> useGlobal;
                    if (useGlobal == 'n' || useGlobal == 'N') {
                        cout << "Enter block size: ";
                        cin >> blockSize;
                    } else {
                        blockSize = globalBlockSize;
                    }
                    int n_i = (lin + blockSize - 1) / blockSize;
                    int n_k = (lin + blockSize - 1) / blockSize;
                    int n_j = (col + blockSize - 1) / blockSize;
                    totalBlocks = n_i * n_k * n_j;
                    elapsed = OnMultBlock(lin, col, blockSize);
                    }
                    break;
                case 4:
                    algorithm = "LineExtParallel";
                    elapsed = OnMultLineExtParallel(lin, col);
                    break;
                case 5:
                    algorithm = "LineIntParallel";
                    elapsed = OnMultLineIntParallel(lin, col);
                    break;
                default:
                    cout << "Invalid option." << endl;
                    break;
            }
            if (papi_enabled) {
                ret = PAPI_stop(EventSet, values);
                if (ret != PAPI_OK)
                    cout << "ERROR: Stop PAPI" << endl;
                printf("L1 DCM: %lld \n", values[0]);
                printf("L2 DCM: %lld \n", values[1]);
                ret = PAPI_reset(EventSet);
                if (ret != PAPI_OK)
                    cout << "FAIL reset" << endl;
            }
            if (op >= 1 && op <= 5)
                PrintOrWriteResults(algorithm, s, blockSize, totalBlocks, elapsed, values[0], values[1]);
        }
    } while(op != 0);
    
    if (papi_enabled) {
        ret = PAPI_remove_event(EventSet, PAPI_L1_DCM);
        if (ret != PAPI_OK)
            cout << "FAIL remove event" << endl;
        ret = PAPI_remove_event(EventSet, PAPI_L2_DCM);
        if (ret != PAPI_OK)
            cout << "FAIL remove event" << endl;
        ret = PAPI_destroy_eventset(&EventSet);
        if (ret != PAPI_OK)
            cout << "FAIL destroy" << endl;
    }
    
    return 0;
}

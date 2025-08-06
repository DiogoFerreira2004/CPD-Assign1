using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.IO;

/*
##################################################################################################
A ideia é termos um programa que faça todos os tipos de multiplicação pedidos + guardar os 
resultados num ficheiro csv ou dar print no terminal dos valores 
Guardar no formato de csv:
algorithm,size,time
##################################################################################################
*/

class Program
{
    static bool writing_to_file = false;
    static bool size_mode = false;
    static string file_name = "metrics_cs/results_cs.csv";

    static int[] sizes = {600, 1000, 1400, 1800, 2200, 2600, 3000};

    static void InitializeMatrices(double[,] matrixA, double[,] matrixB, double[,] matrixC)
    {
        int n = matrixA.GetLength(0);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                matrixA[i, j] = 1.0;
                matrixB[i, j] = (double)(i + 1);
                matrixC[i, j] = 0.0;
            }
        }
    }

    static void CleanMatrices(double[] matrixA, double[] matrixB, double[] matrixC)
    {
        matrixA = null;
        matrixB = null;
        matrixC = null;
    }

    static void PrepareOutputFile()
    {
        string directory = Path.GetDirectoryName(file_name);
        if (!Directory.Exists(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(file_name, "algorithm,size,time\n");

    }

    static void WriteToFile(string algorithm, int size, double time)
    {
        string line = algorithm + "," + size + "," + time + "\n";
        File.AppendAllText(file_name, line);
    }

    static void PrintResults(string algorithm, int size, double time)
    {
        Console.WriteLine(algorithm + " - " + size + " - " + time + "(s)");
    }

    static void PrintOrWriteResults(string algorithm, int size, double time)
    {
        if (writing_to_file)
        {
            WriteToFile(algorithm, size, time);
        }
        else
        {
            PrintResults(algorithm, size, time);
        }
    }

    static void OnMult(int m_ar, int m_br)
    {
        Stopwatch stopwatch = new Stopwatch();
        double temp;
        int i, j, k;

        double[] pha = new double[m_ar * m_ar];
        double[] phb = new double[m_ar * m_ar];
        double[] phc = new double[m_ar * m_ar];

        for (i = 0; i < m_ar; i++)
            for (j = 0; j < m_ar; j++)
                pha[i * m_ar + j] = 1.0;

        for (i = 0; i < m_br; i++)
            for (j = 0; j < m_br; j++)
                phb[i * m_br + j] = (double)(i + 1);

        stopwatch.Start();

        for (i = 0; i < m_ar; i++)
        {
            for (j = 0; j < m_br; j++)
            {
                temp = 0;
                for (k = 0; k < m_ar; k++)
                {
                    temp += pha[i * m_ar + k] * phb[k * m_br + j];
                }
                phc[i * m_ar + j] = temp;
            }
        }

        stopwatch.Stop();

        PrintOrWriteResults("Standard", m_ar, stopwatch.Elapsed.TotalSeconds);
    }

    static void OnMultLine(int m_ar, int m_br)
    {
        double[] pha = new double[m_ar * m_ar];
        double[] phb = new double[m_ar * m_ar];
        double[] phc = new double[m_ar * m_ar];

        for (int i = 0; i < m_ar; i++)
        {
            for (int j = 0; j < m_ar; j++)
            {
                pha[i * m_ar + j] = 1.0;
            }
        }

        for (int i = 0; i < m_br; i++)
        {
            for (int j = 0; j < m_br; j++)
            {
                phb[i * m_br + j] = (double)(i + 1);
            }
        }

        Stopwatch stopwatch = new Stopwatch();
        stopwatch.Start();

        for (int i = 0; i < m_ar; i++)
        {
            for (int k = 0; k < m_ar; k++)
            {
                double r = pha[i * m_ar + k];
                for (int j = 0; j < m_br; j++)
                {
                    phc[i * m_ar + j] += r * phb[k * m_br + j];
                }
            }
        }

        stopwatch.Stop();

        PrintOrWriteResults("Line", m_ar, stopwatch.Elapsed.TotalSeconds);
    }

    static void OnMultBlock(int m_ar, int m_br, int bkSize)
    {
        double[] pha = new double[m_ar * m_ar];
        double[] phb = new double[m_ar * m_ar];
        double[] phc = new double[m_ar * m_ar];

        // Initialize matrices
        for (int i = 0; i < m_ar; i++)
        {
            for (int j = 0; j < m_ar; j++)
            {
                pha[i * m_ar + j] = 1.0;
            }
        }

        for (int i = 0; i < m_br; i++)
        {
            for (int j = 0; j < m_br; j++)
            {
                phb[i * m_br + j] = (double)(i + 1);
            }
        }

        // Block multiplication
        Stopwatch stopwatch = new Stopwatch();
        stopwatch.Start();

        for (int iBlock = 0; iBlock < m_ar; iBlock += bkSize)
        {
            for (int kBlock = 0; kBlock < m_ar; kBlock += bkSize)
            {
                for (int jBlock = 0; jBlock < m_br; jBlock += bkSize)
                {
                    int iMax = Math.Min(iBlock + bkSize, m_ar);
                    int kMax = Math.Min(kBlock + bkSize, m_ar);
                    int jMax = Math.Min(jBlock + bkSize, m_br);
                    
                    for (int i = iBlock; i < iMax; i++)
                    {
                        for (int k = kBlock; k < kMax; k++)
                        {
                            double temp = pha[i * m_ar + k];
                            for (int j = jBlock; j < jMax; j++)
                            {
                                phc[i * m_br + j] += temp * phb[k * m_br + j];
                            }
                        }
                    }
                }
            }
        }

        stopwatch.Stop();

        PrintOrWriteResults($"Block_{bkSize}", m_ar, stopwatch.Elapsed.TotalSeconds);
    }

    static void RunLargeMatrixTests(bool wasWritingToFile = true)
    {
        int[] largeSizes = { 4096, 6144, 8192, 10240 };
        int[] blockSizes = { 128, 256, 512 };
        
        Console.WriteLine("Block multiplication tests for large matrices (4096-10240)");
        
        writing_to_file = true;
        
        try
        {
            foreach (int size in largeSizes)
            {
                Console.WriteLine($"Testing matrix size: {size}x{size}");
                
                // testar todos os tamanhos de bloco
                foreach (int blockSize in blockSizes)
                {
                    Console.WriteLine($"  Block multiplication with block size {blockSize}...");
                    try
                    {
                        OnMultBlock(size, size, blockSize);
                    }
                    catch (OutOfMemoryException)
                    {
                        Console.WriteLine($"  Out of memory for Block size {blockSize}!");
                    }
                    
                    // forçar garbage collection para minimizar pressão na memória
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                }
                
                GC.Collect();
                GC.WaitForPendingFinalizers();
            }
        }
        catch (OutOfMemoryException)
        {
            Console.WriteLine("Out of memory error!");
        }
        finally
        {
            writing_to_file = wasWritingToFile;
        }
    }

    static void Main(string[] args)
    {
        int op;
        do
        {
            Console.WriteLine("1. Multiplication");
            Console.WriteLine("2. Line Multiplication");
            Console.WriteLine("3. Block Multiplication");
            Console.WriteLine("4. All Multiplications");
            Console.WriteLine("8. Toggle Size Mode");
            Console.WriteLine("9. Toggle Save/Display Mode");
            Console.WriteLine("0. Exit");
            Console.Write("Selection?: ");

            if (writing_to_file)
            {
                Console.WriteLine("Result Display Mode: CSV File");
            }
            else
            {
                Console.WriteLine("Result Display Mode: Console");
            }
            if (size_mode)
            {
                Console.WriteLine("Size Mode: All Sizes");
            }
            else
            {
                Console.WriteLine("Size Mode: Fixed");
            }

            op = int.Parse(Console.ReadLine());
            if (op == 0) break;

            if (op == 9)
            {
                writing_to_file = !writing_to_file;
                if (writing_to_file)
                {
                    PrepareOutputFile();
                }
                continue;
            }

            if (op == 8)
            {
                size_mode = !size_mode;
                continue;
            }

            if (!size_mode)
            {
                Console.Write("Dimensions: lins=cols ? ");
                int singleLin = int.Parse(Console.ReadLine());
                int col = singleLin;
                switch (op)
                {
                    case 1:
                        OnMult(singleLin, col);
                        break;
                    case 2:
                        OnMultLine(singleLin, col);
                        break;
                    case 3:
                        Console.Write("Block Size? ");
                        int blockSize = int.Parse(Console.ReadLine());
                        OnMultBlock(singleLin, col, blockSize);
                        break;
                    case 4:
                        bool prevWritingMode = writing_to_file;
                        writing_to_file = true;
                        PrepareOutputFile();
                        
                        Console.WriteLine($"Running all multiplication tests for size {singleLin}x{singleLin}...");
                        OnMult(singleLin, col);
                        OnMultLine(singleLin, col);
                        
                        int[] blockSizes = { 128, 256, 512 };
                        foreach (int bs in blockSizes)
                        {
                            OnMultBlock(singleLin, col, bs);
                        }
                        
                        Console.Write("Run large matrix tests (4096-10240)? This may take a long time (y/n): ");
                        string response = Console.ReadLine().Trim().ToLower();
                        if (response == "y" || response == "yes")
                        {
                            RunLargeMatrixTests(prevWritingMode);
                        }
                        else
                        {
                            writing_to_file = prevWritingMode;
                        }
                        break;
                    default:
                        Console.WriteLine("Invalid Option");
                        break;
                }
            }
            else
            {
                sizes = new int[] {600, 1000, 1400, 1800, 2200, 2600, 3000};
                foreach (int lin in sizes)
                {
                    int col = lin;
                    switch (op)
                    {
                        case 1:
                            OnMult(lin, col);
                            break;
                        case 2:
                            OnMultLine(lin, col);
                            break;
                        case 3:
                            Console.Write("Block Size? ");
                            int blockSize = int.Parse(Console.ReadLine());
                            OnMultBlock(lin, col, blockSize);
                            break;
                        case 4:
                            bool prevWritingMode = writing_to_file;
                            writing_to_file = true;
                            PrepareOutputFile();
                            int[] blockSizes = { 128, 256, 512 };
                            
                            Console.WriteLine("Running comprehensive tests for all matrix sizes...");
                            
                            foreach (int s in sizes)
                            {
                                int col2 = s;
                                Console.WriteLine($"Testing matrix size {s}x{col2}...");
                                
                                OnMult(s, col2);
                                OnMultLine(s, col2);
                                
                                foreach (int bs in blockSizes)
                                {
                                    OnMultBlock(s, col2, bs);
                                }
                                
                                GC.Collect();
                                GC.WaitForPendingFinalizers();
                            }
                            
                            RunLargeMatrixTests(prevWritingMode);
                            break;
                        default:
                            Console.WriteLine("Invalid Option");
                            break;
                    }
                }
            }

        } while (op != 0);
    }
}
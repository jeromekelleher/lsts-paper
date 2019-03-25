import java.io.*; 


/* Based on the source of Beagle 4.1,
 * https://faculty.washington.edu/browning/beagle/b4_1.html.
 * See LSHapBaum.java implementation */

public class ForwardMatrix {
    public final int[][] panel;
    public final int numHaplotypes;
    public final int numSites;
    public final double mutationRate;
    public final double recombinationRate;

    public ForwardMatrix(int[][] panel, double recombinationRate, double mutationRate) {
        this.panel = panel;
        this.numHaplotypes = panel.length;
        this.numSites = panel[0].length;
        this.recombinationRate = recombinationRate;
        this.mutationRate = mutationRate;
    }

    public void printState() {
        System.out.printf("n = %d m = %d\n", numHaplotypes, numSites);
        System.out.println("Panel = ");
        for (int j = 0; j < numHaplotypes; j++) {
            for (int k = 0; k < numSites; k++) {
                System.out.printf("%d", panel[j][k]);
            }
            System.out.println();
        }
    }

//     public double[][] getMatrix(int[] haplotype) {
//         int n = numHaplotypes;
//         double probRec = recombinationRate;
//         double probNoRec = 1.0f - probRec;
//         double noErrProb = 1.0f - mutationRate;
//         double errProb = mutationRate;
//         double shift = probRec/n;
//         double[][] fwdVal = new double[numSites][numHaplotypes];
//         double lastSum = 1.0f;
//         for (int m  = 0; m < numSites - 1; ++m) {
//             // System.out.printf("m = %d\n", m);
//             int prev = m;
//             int next = m + 1;
//             double sum = 0.0f;
//             double scale = probNoRec / lastSum;
//             int a = haplotype[m];
//             for (int h = 0; h < n; ++h) {
//                 // System.out.printf("\th = %d\n", h);
//                 double em = (a == panel[h][m]) ? noErrProb : errProb;
//                 fwdVal[next][h] = m == 0 ? em : em * (scale * fwdVal[prev][h] + shift);
//                 sum += fwdVal[next][h];
//             }
//             lastSum = sum;
//         }
//         return fwdVal;
//     }
//
        
    // No scaling here.
    // Should be getting the same values, but looks like we have a slightly
    // different probability model? Also, why is the first locus set to 0?
    public double[][] getMatrix(int[] haplotype) {
        int n = numHaplotypes;
        double probRec = recombinationRate;
        double probNoRec = 1.0f - probRec;
        double noErrProb = 1.0f - mutationRate;
        double errProb = mutationRate;
        double shift = probRec/n;
        double[][] fwdVal = new double[numSites][numHaplotypes];
        for (int m  = 0; m < numSites - 1; ++m) {
            // System.out.printf("m = %d\n", m);
            int prev = m;
            int next = m + 1;
            double scale = probNoRec;
            int a = haplotype[m];
            for (int h = 0; h < n; ++h) {
                // System.out.printf("\th = %d\n", h);
                double em = (a == panel[h][m]) ? noErrProb : errProb;
                fwdVal[next][h] = m == 0 ? em : em * (scale * fwdVal[prev][h] + shift);
            }
        }
        return fwdVal;
    }


    public double[][] rescaleMatrix(double[][] matrix)
    {
        double[][] rescaled = new double[numSites][numHaplotypes];
        double probRec = recombinationRate;
        double probNoRec = 1.0f - probRec;
        double lastSum = 1.0;
        for (int m  = 0; m < numSites - 1; ++m) {
            int prev = m;
            int next = m + 1;
            double sum = 0.0f;
            double scale = probNoRec / lastSum;
            for (int h = 0; h < numHaplotypes; ++h) {
                sum += matrix[next][h];
                rescaled[next][h] = matrix[next][h] / scale;
            }
            lastSum = sum;
        }
        return rescaled;
    }
    

    // private void setForwardValues(int start, int end, int hap) {
    //     double lastSum = 1.0f;
    //     for (int m=start; m<end; ++m) {
    //         double probRec = impData.pRecomb(m);
    //         double probNoRec = 1.0f - probRec;
    //         double noErrProb = impData.noErrProb(m);
    //         double errProb = impData.errProb(m);
    //         double shift = probRec/n;
    //         double scale = probNoRec/lastSum;
    //         int prev = currentIndex();
    //         int next = nextIndex();
    //         double sum = 0.0f;
    //         fwdValueIndex2Marker[next] = m;
    //         int a = impData.targetAllele(m, hap);
    //         for (int h=0; h<n; ++h) {
    //             double em = (a == impData.refAllele(m, h)) ? noErrProb : errProb;
    //             fwdVal[next][h] = m==0 ? em : em*(scale*fwdVal[prev][h] + shift);
    //             sum += fwdVal[next][h];
    //         }
    //         lastSum = sum;
    //     }
    // }




    public static int[][] readPanel(String filename) throws Exception {

        File file = new File(filename);
        BufferedReader br = new BufferedReader(new FileReader(file)); 
        String line; 
        line = br.readLine();
        String[] tokens = line.split(" ");
        int n = Integer.parseInt(tokens[0]);
        int m = Integer.parseInt(tokens[1]);
        int[][] panel  = new int[n][m];

        int j = 0;
        while ((line = br.readLine()) != null) {
            for (int k = 0; k < m; k++) {
                panel[j][k] = Integer.parseInt(line.substring(k, k + 1));
            }
            j++;
        }
        return panel;
    }

    public static void main(String[] args) throws Exception {

        int[][] panel = ForwardMatrix.readPanel(args[0]);
        ForwardMatrix fm = new ForwardMatrix(panel, 0.125, 0.0125);
        fm.printState();
        int h[] = new int[fm.numSites];
        for (int j = 0; j < fm.numSites; j++) {
            h[j] = 0;
        }
        double[][] matrix = fm.getMatrix(h);
        double [][] unscaled = fm.rescaleMatrix(matrix);

        for (int j = 0; j < fm.numHaplotypes; j++) {
            // System.out.printf("%d\t", l);
            for (int l = 0; l < fm.numSites; l++) {
                System.out.printf ("%f ", matrix[l][j]);
            }
            System.out.println();
        }

        // System.out.println("Unscaled");
        // for (int j = 0; j < fm.numHaplotypes; j++) {
        //     // System.out.printf("%d\t", l);
        //     for (int l = 0; l < fm.numSites; l++) {
        //         System.out.printf ("%f ", unscaled[l][j]);
        //     }
        //     System.out.println();
        // }
        //
    }



}

package edu.nchu.app;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.*;
import org.apache.spark.broadcast.Broadcast;

import scala.Tuple2;
import scala.Tuple3;

public class kmeans {
    // input parameter
    static String dataset_path;
    static String output_path;
    static int num_k;
    static int num_iter;
    static int num_run;

    static Double[][] dataset;
	static Vector<Double> max_value;
	static Vector<Double> min_value;
    static int num_dimension;
	static int num_data;
    static FileWriter fw;
    static double best_objectvalue;
    // time parameter
    static long start_t,total_t,end_t,map_t,reduce_t;
    // spark
	static JavaSparkContext sc;
	static JavaRDD datasetRDD;
    static Broadcast<Double[][]> bc_dataset;


	public static void main( String[] args ) throws IOException
	{
        // 1.input parameter
        dataset_path = args[0];
        num_iter = Integer.valueOf( args[1] );
        num_k = Integer.valueOf( args[2] );
        output_path = args[3];
        num_run = Integer.valueOf( args[4] );

        // 2.start spark
        SparkConf conf = new SparkConf();
		sc = new JavaSparkContext(conf);

        // 3.read dataset,set output
        readDataset();
        fw = new FileWriter( output_path );

        total_t = System.currentTimeMillis();
        for( int run=0; run<num_run; run++ )
        {
            best_objectvalue = Double.MAX_VALUE;
            start();
        }
        fw.write( System.currentTimeMillis() - total_t + "ms" );
        fw.flush();
        fw.close();
        
    }

    static void start() throws IOException
    {
        // 4.init
        // center: (1)index, (2)position
        // sol: (1)cluster (2)position (3)min_dist
        // dataset: (1)position
        Double[][] center;
        JavaPairRDD<Integer, Tuple2<Double[], Double>> sol;

        center = create_center( num_k );

        // 5.do until terminate
        for( int iter=0; iter<num_iter; iter++ )
		{
            start_t = System.currentTimeMillis();

            center = spark_kmeans( center );

            end_t = System.currentTimeMillis();
            print_best( iter + 1 );
		}
    }
    static Double[][] spark_kmeans( Double[][] center )
    {
        JavaPairRDD<Integer, Tuple2<Double[], Double>> sol;
        Broadcast<Double[][]> bc_dataset = sc.broadcast( dataset );
        Broadcast<Double[][]> bc_center = sc.broadcast( center );

        // (1)position (2)center
        List<Tuple2<Double[],Double[][]>> dataset_add_center = new ArrayList();
        for( int i=0; i<num_data; i++ )
            dataset_add_center.add( new Tuple2< Double[],Double[][] >(dataset[i], center) );
        JavaRDD dataset_rdd = sc.parallelize( dataset_add_center );

        PairFunction< Integer, Integer, Tuple2<Integer,Double>> f1 = new PairFunction
                    < Integer, Integer, Tuple2<Integer,Double>> ()
        {
            public Tuple2<Integer, Tuple2<Integer, Double>> call( Integer arg0 ) throws Exception 
            {
                double min_dist = compute_dist( bc_dataset.getValue()[arg0], bc_center.getValue()[0] );
                int k = 0;
                for( int i=1; i<bc_center.getValue().length; i++ )
                {
                    double dist = compute_dist( bc_dataset.getValue()[arg0], bc_center.getValue()[i] );
                    if( min_dist > dist )
                    {
                        min_dist = dist;
                        k = i;
                    }
                }
                if( System.currentTimeMillis() > map_t )
                    map_t = System.currentTimeMillis();
                return new Tuple2<Integer,Tuple2<Integer, Double>>( k, new Tuple2<Integer, Double>( arg0, min_dist ) );
            }
        };
		Function<Tuple2<Integer,Double>,Tuple3<Double[],Double,Integer>> combine_f1 = new Function
				<Tuple2<Integer,Double>,Tuple3<Double[],Double,Integer>>()
        {
            public Tuple3<Double[],Double,Integer> call(
                    Tuple2<Integer, Double> arg0) throws Exception {
                Double[] position = bc_dataset.value()[arg0._1()].clone();
                Double min_dist = arg0._2();
                Integer count = 1;
                return new Tuple3<Double[],Double,Integer>(position,min_dist,count);
            }
        };
		Function2<Tuple3<Double[],Double,Integer>,Tuple2<Integer,Double>,Tuple3<Double[],Double,Integer>> combine_f2 = new Function2
				 <Tuple3<Double[],Double,Integer>,Tuple2<Integer,Double>,Tuple3<Double[],Double,Integer>>()
        {
            public Tuple3<Double[],Double,Integer> call( Tuple3<Double[],Double,Integer> arg0, Tuple2<Integer, Double> arg1) throws Exception 
            {
                Double[] position = arg0._1();
                Double[] position1 = bc_dataset.value()[arg1._1()].clone();
                Double min_dist = arg0._2() + arg1._2();
                Integer count = arg0._3() + 1;
                for( int i=0; i<position.length; i++ )
                    position[i] += position1[i];
                return new Tuple3<Double[], Double, Integer>(position,min_dist,count);
            }
        };
		Function2<Tuple3<Double[],Double,Integer>,Tuple3<Double[],Double,Integer>,Tuple3<Double[],Double,Integer>> combine_f3 = new Function2
				 <Tuple3<Double[],Double,Integer>,Tuple3<Double[],Double,Integer>,Tuple3<Double[],Double,Integer>>()
        {
            public Tuple3<Double[], Double, Integer> call(
                    Tuple3<Double[], Double, Integer> arg0,
                    Tuple3<Double[], Double, Integer> arg1)
                    throws Exception {
                Double[] position = arg0._1();
                Double[] position1 = arg1._1();
                Double min_dist = arg0._2() + arg1._2();
                Integer count = arg0._3() + arg1._3();
                for( int i=0; i<position.length; i++ )
                    position[i] += position1[i];
                return new Tuple3<Double[], Double, Integer>(position,min_dist,count);
            }
        };

		Function<Tuple2<Integer,Tuple3<Double[],Double,Integer>>,Tuple2<Integer,Tuple2<Double[],Double>>> devide_count = new Function
				<Tuple2<Integer,Tuple3<Double[],Double,Integer>>,Tuple2<Integer,Tuple2<Double[],Double>>>()
        {
            public Tuple2<Integer, Tuple2<Double[], Double>> call(Tuple2<Integer, Tuple3<Double[], Double, Integer>> arg0)
                    throws Exception {
                Double[] position = arg0._2()._1();
                Double min_dist = arg0._2()._2();
                Integer count = arg0._2()._3();
                for( int i=0; i<position.length; i++ )
                    position[i] /= count;
                Tuple2<Double[], Double> T2 = new Tuple2<Double[], Double>(position,min_dist);
                return new Tuple2<Integer, Tuple2<Double[], Double>>( arg0._1(), T2 );
            }
        };

        Integer[] array = new Integer[num_data];
		for( int i=0; i<array.length; i++ )
			array[i] = i;
        List<Tuple2<Integer, Tuple2<Double[], Double>>> list = sc.parallelize(Arrays.asList(array)).mapToPair(f1).combineByKey(combine_f1,combine_f2,combine_f3).map(devide_count).collect();
        // sc.parallelize(Arrays.asList(array)).mapToPair(f1).combineByKey(combine_f1,combine_f2,combine_f3).collect();
		double SSE = 0;
		for( int i=0; i<list.size(); i++ )
		{
			center[list.get(i)._1()] = list.get(i)._2()._1();
			SSE += list.get(i)._2()._2();
		}
		for( int i=0; i<num_k; i++ )
			if(center[i].length == 0)
				for( int j=0; j<num_dimension; j++ )
                    center[i] = dataset[(int)random(0, num_data)].clone();

        reduce_t = System.currentTimeMillis();
		
        if( best_objectvalue > SSE )
            best_objectvalue = SSE;
        return center;
    }
    static void readDataset() throws IOException
	{
		Vector<Vector<Double>> dataset_tmp = new Vector<Vector<Double>>();
        max_value = new Vector<Double>();
        min_value = new Vector<Double>();

		FileReader fr = new FileReader( dataset_path );
		BufferedReader br = new BufferedReader(fr);
		while (br.ready()) {
			String[] tokens = br.readLine().split(",");
			Vector<Double> tmp = new Vector<Double>();
			for( int i=0; i<tokens.length; i++ )
				tmp.add(Double.parseDouble(tokens[i]));
			if( tmp.size()!=0 )
				dataset_tmp.add(tmp);
			
			for( int i=0; i<tmp.size(); i++ )
			{
				if( i > max_value.size()-1 )
					max_value.add( tmp.get(i) );
				else if( tmp.get(i) > max_value.get(i) )
					max_value.set(i, tmp.get(i));
				
				if( i > min_value.size()-1 )
					min_value.add( tmp.get(i) );
				else if( tmp.get(i) < min_value.get(i) )
					min_value.set(i, tmp.get(i));
			}
		}
		fr.close();
		num_dimension = max_value.size();
		num_data = dataset_tmp.size();
        dataset = new Double[num_data][num_dimension];
        for( int i=0; i<num_data; i++ )
			for( int j=0; j<num_dimension; j++ )
				dataset[i][j] = dataset_tmp.get(i).get(j);
	}
    static Double[][] create_center( int num_k )
	{
        Double[][] center = new Double[num_k][num_dimension];
		for( int i=0; i<num_k; i++ )
		{
			int rand = (int)random( 0, num_data );
			center[i] = dataset[rand].clone();
		}
        return center;
	}
    static double random( double min, double max )
	{
		return Math.random()*(max-min) + min;
	}
    static double compute_dist( Double[] data, Double[] center )
	{
		double sum = 0;
		for( int i=0; i<data.length; i++ )
			sum += Math.pow(center[i] - data[i],2);
		return Math.sqrt(sum);
	}
    static void print_best( int iter ) throws IOException
    {
        fw.write( iter + "," );
        fw.write( best_objectvalue + "," );
        fw.write( (map_t-start_t) + "," ); // map time
        fw.write( (reduce_t-map_t) + "," ); // reduce time
        fw.write( (end_t-start_t) + "," ); // this iteration time
        fw.write( "\n" );
        fw.flush();
    }
}

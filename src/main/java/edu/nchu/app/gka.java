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

public class gka {
    // input parameter
    static String dataset_path;
    static String output_path;
    static int num_k;
    static int num_iter;
    static int num_run;
    // gka
    static int num_cms;
	static double Pm;
	static double Cm;

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
    // gka
    static Integer[][] cms_sol;
	static Double[] cms_objectvalue;
	static List<Tuple2<Integer[],Double>> cms_list;
	static JavaPairRDD< Integer[], Double > cms_sol_rdd;

	public static void main( String[] args ) throws IOException
	{
        // 1.input parameter
        dataset_path = args[0];
        num_iter = Integer.valueOf( args[1] );
        num_k = Integer.valueOf( args[2] );
        output_path = args[3];
        num_run = Integer.valueOf( args[4] );
        num_cms = Integer.valueOf( args[5] );
        Pm = Double.parseDouble( args[6] );
        Cm = Double.parseDouble( args[7] );

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
        cms_list = create_cms( num_cms );


        // 5.do until terminate
		for( int iter=0; iter<num_iter; iter++  )
		{
			start_t = System.currentTimeMillis();

			cms_sol_rdd = spark_selection(); //selction    
		    cms_list = new ArrayList( spark_transition_fitness( cms_sol_rdd ));
			reduce_t = System.currentTimeMillis();
		    for( int i=0; i<num_cms; i++ )
                if( best_objectvalue > cms_list.get(i)._2() )
                    best_objectvalue = cms_list.get(i)._2();

			end_t = System.currentTimeMillis();
			print_best( iter + 1 ); //print
		}
    }
    static List<Tuple2<Integer[],Double>> spark_transition_fitness( JavaPairRDD< Integer[], Double > cms_sol_rdd )
    {
		Broadcast<Double[][]> bc_dataset = sc.broadcast( dataset );
		Broadcast<Integer> bc_num_k = sc.broadcast( num_k );
		Broadcast<Integer> bc_num_data = sc.broadcast( num_data );
		Broadcast<Integer> bc_num_dimension = sc.broadcast( num_dimension );
		Broadcast<Double> bc_Pm = sc.broadcast( Pm );
		Broadcast<Double> bc_Cm = sc.broadcast( Cm );

		PairFunction<Tuple2<Integer[], Double>, Integer[], Double> transit_fitness = new PairFunction
					<Tuple2<Integer[], Double>, Integer[], Double>()
		{
			public Tuple2<Integer[], Double> call(Tuple2<Integer[], Double> input) throws Exception 
            {
				// parameter from broadcast
				int num_k = (int)bc_num_k.getValue();
				int num_data = (int)bc_num_data.getValue();
				int num_dimension = (int)bc_num_dimension.getValue();
				double Pm = (double)bc_Pm.getValue();
				double Cm = (double)bc_Cm.getValue();
				// parameter
				Integer[] sol = input._1();
				double objectvalue = 1;
				int index = -1;
				Double[] dist = new Double[num_data];
				// 1.kmeans
				int[] count = new int[num_k];
				Double[][] center = new Double[num_k][num_dimension];
				for( int i=0; i<num_k; i++ )
				{
					count[i] = 0;
					for( int j=0; j<num_dimension; j++ )
						center[i][j] = 0.0;
				}
				// 1.1  update
				for( int i=0; i<num_data; i++ )
				{
					count[sol[i]]++;
					for( int j=0; j<num_dimension; j++ )
						center[sol[i]][j] += bc_dataset.getValue()[i][j];
				}
				for( int i=0; i<num_k; i++ )
					for( int j=0; j<num_dimension; j++ )
						center[i][j] /= count[i];
				// 1.2 assign
				for( int i=0; i<num_data; i++ )
				{
					double min_dist = compute_dist( bc_dataset.getValue()[i], center[0] );
					int k = 0;
					for( int j=1; j<center.length; j++ )
					{
						double tmp_dist = compute_dist( bc_dataset.getValue()[i], center[j] );
						if( min_dist > tmp_dist )
						{
							min_dist = tmp_dist;
							k = j;
						}
					}
					sol[i] = k;
					dist[i] = min_dist;
				}
				// 2.mutation
				if( Math.random() < Pm )
				{
					double dist_max = 0.0;
					double total = 0.0;
					Double[] roulette_wheel = new Double[num_data];
					count = new int[num_k];
					center = new Double[num_k][num_dimension];
					for( int i=0; i<num_k; i++ )
					{
						count[i] = 0;
						for( int j=0; j<num_dimension; j++ )
							center[i][j] = 0.0;
					}
					// 2.1 compute distance
					for( int i=0; i<num_data; i++ )
					{
						count[sol[i]]++;
						for( int j=0; j<num_dimension; j++ )
							center[sol[i]][j] += bc_dataset.getValue()[i][j];
					}
					for( int i=0; i<num_k; i++ )
						for( int j=0; j<num_dimension; j++ )
							center[i][j] /= count[i];
					for( int i=0; i<num_data; i++ )
					{
						double min_dist = compute_dist( bc_dataset.getValue()[i], center[0] );
						int k = 0;
						for( int j=1; j<center.length; j++ )
						{
							double tmp_dist = compute_dist( bc_dataset.getValue()[i], center[j] );
							if( min_dist > tmp_dist )
							{
								min_dist = tmp_dist;
								k = j;
							}
						}
						dist[i] = min_dist;
						if( dist[i] > dist_max )
							dist_max = dist[i];
					}
					// 2.2 make roulette wheel
					for( int i=0; i<num_data; i++ )
					{
						roulette_wheel[i] = Cm * dist_max - dist[i];
						total += roulette_wheel[i];
					}
					roulette_wheel[0] = roulette_wheel[0]/total;
					for( int i=1; i<num_data; i++ )
						roulette_wheel[i] = roulette_wheel[i]/total + roulette_wheel[i-1];
					// 2.3 choose and change
					double f = random(0, roulette_wheel[roulette_wheel.length-1]);
					if (f < roulette_wheel[0])
					{
						sol[0] = (int)random(0, num_k);
						index = 0;
					}
					else if( f == roulette_wheel[roulette_wheel.length-1] )
					{
						sol[num_data-1] = (int)random(0, num_k);
						index = num_data-1;
					}
					else 
					{
						for (int k = 1; k < roulette_wheel.length; k++)      
							if (f >= roulette_wheel[k-1] && f < roulette_wheel[k]) 
							{
								sol[k] = (int)random(0, num_k);
								index = k;
								break;
							}	
					}
				}
				// 3. evaluation
				objectvalue = 0;
				for( int i=0; i<num_data; i++ )
					objectvalue += dist[i];
				if( index != -1 )
					objectvalue = objectvalue - dist[index] + compute_dist( bc_dataset.getValue()[index], center[sol[index]] );

				if( System.currentTimeMillis() > map_t )
                    map_t = System.currentTimeMillis();
				return new Tuple2<Integer[], Double>( sol, objectvalue );
			}
		};
		
		return cms_sol_rdd.mapToPair(transit_fitness).collect();
    }
    static JavaPairRDD< Integer[], Double > spark_selection() throws IOException
    {
		Integer[][] cms_sol = new Integer[num_cms][num_data]; //染色體
		double[] cms_objectvalue = new double[num_cms]; //sse
		// copy cms from Tuple2 取出染色體
		for( int i=0; i<num_cms; i++ )
		{
			Tuple2<Integer[], Double> tmp_Tuple2 = cms_list.get(i);
			cms_sol[i] = tmp_Tuple2._1();
			cms_objectvalue[i] = tmp_Tuple2._2();
		}
		// Build RoultteWheel 建立羅盤
		double total = 0;
		Double[] RouletteWheel = new Double[num_cms];
		
		for( int i=0; i<cms_objectvalue.length; i++ )
			total += 1/cms_objectvalue[i];
			
		RouletteWheel[0] = (1/cms_objectvalue[0]) / total;
		for( int i=1; i<cms_objectvalue.length; i++ )
			RouletteWheel[i] = (1/cms_objectvalue[i]) / total + RouletteWheel[i-1];
		// 射飛鏢
		cms_list = new ArrayList<Tuple2<Integer[],Double>>();
		for( int i=0; i<num_cms; i++ )
		{
			double f = Math.random();
			int index = -1;
			Tuple2<Integer[],Double> tmp_cms = new Tuple2<Integer[],Double>( cms_sol[0], cms_objectvalue[0] );
			if (f < RouletteWheel[0])
			{
				tmp_cms = new Tuple2<Integer[],Double>( cms_sol[0], cms_objectvalue[0] );
				index = 0;
			}
		    else if( f >= RouletteWheel[num_cms-1] )
			{
		    	tmp_cms = new Tuple2<Integer[],Double>( cms_sol[num_cms-1], cms_objectvalue[num_cms-1] );
				index = num_cms-1;
			}
			else 
			{
		   		for (int j = 1; j < num_cms; j++)      
					if (f >= RouletteWheel[j-1] && f < RouletteWheel[j]) 
					{
						tmp_cms = new Tuple2<Integer[],Double>( cms_sol[j], cms_objectvalue[j] );
						index = j;
			    		break;
					}
			}
			cms_list.add( tmp_cms );
		}
        return JavaPairRDD.fromJavaRDD( sc.parallelize(cms_list) );
    }
    static List<Tuple2<Integer[],Double>> create_cms( int num_cms ) throws IOException
    {
        List<Tuple2<Integer[],Double>> cms_list = new ArrayList<Tuple2<Integer[],Double>>();
        
		// create cms
		for( int i=0; i<num_cms; i++ )
			cms_list.add( new Tuple2<Integer[],Double>( create_sol(num_k),0.0 ) );
		// spark evaluate
		Broadcast<Double[][]> bc_dataset = sc.broadcast( dataset );
		Broadcast<Integer> bc_num_k = sc.broadcast( num_k );
		Broadcast<Integer> bc_num_data = sc.broadcast( num_data );
		Broadcast<Integer> bc_num_dimension = sc.broadcast( num_dimension );


		Function<Tuple2<Integer[],Double>,Tuple2<Integer[],Double>> evaluate = new Function
				<Tuple2<Integer[],Double>,Tuple2<Integer[],Double>>()
		{
			public Tuple2<Integer[],Double> call( Tuple2<Integer[],Double> input ) throws Exception 
			{
				// parameter from broadcast
				int num_k = (int)bc_num_k.getValue();
				int num_data = (int)bc_num_data.getValue();
				int num_dimension = (int)bc_num_dimension.getValue();

				Integer[] sol = input._1();
				double objectvalue = 0;
				// init count
				int[] count = new int[num_k];
				// init center
				Double[][] center = new Double[num_k][num_dimension]; 
				for ( int i=0; i<num_k; i++ ) 
					for( int j=0; j<num_dimension; j++ )
						center[i][j] = 0.0;
				// update center
				for( int i=0; i<num_data; i++ )
				{
					count[sol[i]]++;
					for( int j=0; j<num_dimension; j++ )
						center[sol[i]][j] += bc_dataset.getValue()[i][j];
				}

				for( int i=0; i<num_k; i++ )
					for( int j=0; j<num_dimension; j++ )
						center[i][j] /= count[i];

				for( int i=0; i<num_data; i++ )
				{
					double min_dist = compute_dist( bc_dataset.getValue()[i], center[0] );
					int k = 0;
					for( int j=1; j<center.length; j++ )
					{
						double dist = compute_dist( bc_dataset.getValue()[i], center[j] );
						if( min_dist > dist )
						{
							min_dist = dist;
							k = j;
						}
					}
					objectvalue += min_dist;
				}
				return new Tuple2<Integer[],Double>( sol,objectvalue );
			}
		};
		cms_list = sc.parallelize(cms_list).map(evaluate).collect();

        // for( int i=0; i<num_cms; i++ )
		// {
		// 	for( int j=0; j<cms_list.get(i)._1().length; j++ )
		// 		fw.write( cms_list.get(i)._1()[j] + "," );
		// 	fw.write( "  " + cms_list.get(i)._2() + "\n" );
		// 	fw.flush();
		// }

        return cms_list;
    }
    static Integer[] create_sol( int num_k )
	{
		Integer[] sol = new Integer[num_data];
		for( int i=0; i<num_data; i++ )
			sol[i] = (int)random(0,num_k);
		return sol;
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
    static double random( double min, double max )
	{
		return Math.random()*(max-min) + min;
	}
}

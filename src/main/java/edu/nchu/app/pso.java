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
import scala.Tuple5;

public class pso {
    // input parameter
    static String dataset_path;
    static String output_path;
    static int num_k;
    static int num_iter;
    static int num_run;
    // pso
    static int num_particle;
    static double learning_c1, learning_c2;
    static double w_max, w_min;

    static double w_now;
    static Double[] v_max, v_min;
    static Double[] particle_objectvalue;
    static Double[] local_best_objectvalue;
    static Double[][][] particle_sol;
    static Double[][][] local_best_sol;
    static Double[][][] particle_v;

    // time parameter
    static long start_t,total_t,end_t,map_t,reduce_t;
    
    static Double[][] dataset;
	static Double[] max_value;
	static Double[] min_value;
    static int num_dimension;
	static int num_data;
    static FileWriter fw;
    static Double[][] best_sol;
    static double best_objectvalue;
    
    // spark
	static JavaSparkContext sc;
	static JavaRDD datasetRDD;

	public static void main( String[] args ) throws IOException
	{
        // 1.input parameter
        dataset_path = args[0];
        num_iter = Integer.valueOf( args[1] );
        num_k = Integer.valueOf( args[2] );
        output_path = args[3];
        num_run = Integer.valueOf( args[4] );
        num_particle = Integer.valueOf( args[5] );
        learning_c1 = Double.parseDouble( args[6] );
        learning_c2 = Double.parseDouble( args[7] );
        w_max = Double.parseDouble( args[8] );
        w_min = Double.parseDouble( args[9] );

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
        List<Tuple5<Double[][],Double,Double[][],Double,Double[][]>> particle;
        particle = init();
        for( int iter=0; iter<num_iter; iter++  )
		{
            start_t = System.currentTimeMillis();

            w_now = w_max - (w_max-w_min) / num_iter * iter;
            particle = pso_transit_evaluate( particle );
            for( int i=0; i<num_particle; i++ )
            {
                if( best_objectvalue > particle.get(i)._4() )
                {
                    best_sol = particle.get(i)._3();
                    best_objectvalue = particle.get(i)._4();
                }
            }

            end_t = System.currentTimeMillis();
            print_best( iter+1 );
        }
    }
    static List<Tuple5<Double[][],Double,Double[][],Double,Double[][]>> pso_transit_evaluate( List<Tuple5<Double[][],Double,Double[][],Double,Double[][]>> particle ) throws IOException
    {
        Broadcast<Double[][]> bc_dataset = sc.broadcast( dataset );
		Broadcast<Integer> bc_num_k = sc.broadcast( num_k );
        Broadcast<Double> bc_learning_c1 = sc.broadcast( learning_c1 );
        Broadcast<Double> bc_learning_c2 = sc.broadcast( learning_c2 );
        Broadcast<Double[]> bc_v_max = sc.broadcast( v_max );
        Broadcast<Double[]> bc_v_min = sc.broadcast( v_min );
        Broadcast<Double[][]> bc_best_sol = sc.broadcast( best_sol );
        Broadcast<Double> bc_w_now = sc.broadcast( w_now );

        Function<Tuple5<Double[][],Double,Double[][],Double,Double[][]>,Tuple5<Double[][],Double,Double[][],Double,Double[][]>> transit = new Function
                <Tuple5<Double[][],Double,Double[][],Double,Double[][]>,Tuple5<Double[][],Double,Double[][],Double,Double[][]>>()
        {
            public Tuple5<Double[][],Double,Double[][],Double,Double[][]> call( Tuple5<Double[][],Double,Double[][],Double,Double[][]> input ) throws Exception 
			{
                int num_k = bc_num_k.getValue();
                int num_dimension = bc_dataset.getValue()[0].length;
                

                Double[][] sol = input._1();
                Double objectvalue = input._2();
                Double[][] local_best_sol = input._3();
                Double local_best_objectvalue = input._4();
                Double[][] v = input._5();

                for( int i=0; i<num_k; i++ )
                    for( int j=0; j<num_dimension; j++ )
                    {
                        v[i][j] *= bc_w_now.getValue();
                        v[i][j] = bc_learning_c1.getValue()*random(0,1)*(local_best_sol[i][j]-sol[i][j]) +
                                  bc_learning_c2.getValue()*random(0,1)*(bc_best_sol.getValue()[i][j]-sol[i][j]);
                        if( v[i][j] > bc_v_max.getValue()[j] )
                            v[i][j] = bc_v_max.getValue()[j];
                        else if( v[i][j] < bc_v_min.getValue()[j] )
                            v[i][j] = bc_v_min.getValue()[j];
                        sol[i][j] += v[i][j];
                    }
                return new Tuple5<Double[][],Double,Double[][],Double,Double[][]>( sol, objectvalue, local_best_sol, local_best_objectvalue, v );
            }
        };
        Function<Tuple5<Double[][],Double,Double[][],Double,Double[][]>,Tuple5<Double[][],Double,Double[][],Double,Double[][]>> evaluate_determin = new Function
                <Tuple5<Double[][],Double,Double[][],Double,Double[][]>,Tuple5<Double[][],Double,Double[][],Double,Double[][]>>()
        {
            public Tuple5<Double[][],Double,Double[][],Double,Double[][]> call( Tuple5<Double[][],Double,Double[][],Double,Double[][]> input ) throws Exception 
			{
                int num_k = bc_num_k.getValue();
                int num_dimension = bc_dataset.getValue()[0].length;
                int num_data = bc_dataset.getValue().length;

                Double[][] sol = input._1();
                Double objectvalue = input._2();
                Double[][] local_best_sol = input._3();
                Double local_best_objectvalue = input._4();
                Double[][] v = input._5();

                // objectvalue
                objectvalue = 0.0;
                for( int i=0; i<num_data; i++ )
                {
                    double min_dist = compute_dist( bc_dataset.getValue()[i], sol[0] );
                    int k = 0;
                    for( int j=1; j<num_k; j++ )
                    {
                        double dist = compute_dist( bc_dataset.getValue()[i], sol[j] );
                        if( min_dist > dist )
                        {
                            min_dist = dist;
                            k = j;
                        }
                    }
                    objectvalue += min_dist;
                }
                if( local_best_objectvalue > objectvalue )
                {
                    local_best_sol = sol.clone();
                    local_best_objectvalue = objectvalue;
                }
                if( System.currentTimeMillis() > map_t )
                    map_t = System.currentTimeMillis();
                return new Tuple5<Double[][],Double,Double[][],Double,Double[][]>( sol, objectvalue, local_best_sol, local_best_objectvalue, v );
            }
        };
        particle = sc.parallelize(particle).map(transit).map(evaluate_determin).collect();
        reduce_t = System.currentTimeMillis();
        return particle;
    }
    static List<Tuple5<Double[][],Double,Double[][],Double,Double[][]>> init() throws IOException
    {
        v_max = new Double[num_dimension];
        v_min = new Double[num_dimension];
        for( int i=0; i<num_dimension; i++ )
		{
			v_max[i] = (max_value[i] - min_value[i])/100;
			v_min[i] = -v_max[i];
		}
        Broadcast<Double[][]> bc_dataset = sc.broadcast( dataset );
		Broadcast<Integer> bc_num_k = sc.broadcast( num_k );
        Broadcast<Double[]> bc_max_value = sc.broadcast( max_value );
        Broadcast<Double[]> bc_min_value = sc.broadcast( min_value );
        Broadcast<Double[]> bc_v_max = sc.broadcast( v_max );
        Broadcast<Double[]> bc_v_min = sc.broadcast( v_min );
       
        Function<Tuple5<Double[][],Double,Double[][],Double,Double[][]>,Tuple5<Double[][],Double,Double[][],Double,Double[][]>> initial = new Function
                <Tuple5<Double[][],Double,Double[][],Double,Double[][]>,Tuple5<Double[][],Double,Double[][],Double,Double[][]>>()
        {
            public Tuple5<Double[][],Double,Double[][],Double,Double[][]> call( Tuple5<Double[][],Double,Double[][],Double,Double[][]> input ) throws Exception 
			{
                int num_k = bc_num_k.getValue();
                int num_data = bc_dataset.getValue().length;
                int num_dimension = bc_dataset.getValue()[0].length;
                Double[][] sol = input._1();
                Double objectvalue = input._2();
                Double[][] v = input._5();
                // sol,v
                for( int i=0; i<num_k; i++ )
                {
                    sol[i] = bc_dataset.getValue()[(int)random(0,num_data)];
                    for( int j=0; j<num_dimension; j++ )                        
                        v[i][j] = random( bc_v_min.getValue()[j], bc_v_max.getValue()[j] );
                }
                    
                // objectvalue
                for( int i=0; i<num_data; i++ )
                {
                    double min_dist = compute_dist( bc_dataset.getValue()[i], sol[0] );
                    int k = 0;
                    for( int j=1; j<num_k; j++ )
                    {
                        double dist = compute_dist( bc_dataset.getValue()[i], sol[j] );
                        if( min_dist > dist )
                        {
                            min_dist = dist;
                            k = j;
                        }
                    }
                    objectvalue += min_dist;
                }
                return new Tuple5<Double[][],Double,Double[][],Double,Double[][]>( sol,objectvalue,sol.clone(),objectvalue,v );
            }
        };

        List<Tuple5<Double[][],Double,Double[][],Double,Double[][]>> particle = new ArrayList<Tuple5<Double[][],Double,Double[][],Double,Double[][]>>();
        for( int h=0; h<num_particle; h++ )
        {
            Double[][] sol = new Double[num_k][num_dimension];
            Double objectvalue = 0.0;
            Double[][] v = new Double[num_k][num_dimension];
            // sol,v
            for( int i=0; i<num_k; i++ )
                for( int j=0; j<num_dimension; j++ )
                {
                    sol[i] = dataset[(int)random(0,num_data)];
                    v[i][j] = random( bc_v_min.getValue()[j], bc_v_max.getValue()[j] );
                }
            // objectvalue
            for( int i=0; i<num_data; i++ )
            {
                double min_dist = compute_dist( bc_dataset.getValue()[i], sol[0] );
                int k = 0;
                for( int j=1; j<num_k; j++ )
                {
                    double dist = compute_dist( bc_dataset.getValue()[i], sol[j] );
                    if( min_dist > dist )
                    {
                        min_dist = dist;
                        k = j;
                    }
                }
                objectvalue += min_dist;
            }
            particle.add( new Tuple5<Double[][],Double,Double[][],Double,Double[][]>(sol,objectvalue,sol.clone(),objectvalue,v) );
        }
        // for( int i=0; i<num_particle; i++ )
        //     particle.add(new Tuple5<Double[][],Double,Double[][],Double,Double[][]>( new Double[num_k][num_dimension], 0.0, new Double[num_k][num_dimension], 0.0, new Double[num_k][num_dimension] ));
        // particle = sc.parallelize( particle ).map(initial).collect();
        
        for( int i=0; i<num_particle; i++ )
            if( best_objectvalue > particle.get(i)._4() )
            {
                best_sol = particle.get(i)._3();
                best_objectvalue = particle.get(i)._4();
            }


        // for( int i=0; i<num_particle; i++ )
        // {
        //     // for( int k=0; k<num_k; k++ )
        //     //     for( int j=0; j<num_dimension; j++ )
        //     //         fw.write( particle.get(i)._1()[k][j] + "," );
        //     fw.write( "   " + particle.get(i)._2() + "   " ); 
        //     // for( int k=0; k<num_k; k++ )
        //     //     for( int j=0; j<num_dimension; j++ )
        //     //         fw.write( particle.get(i)._5()[k][j] + "," ); 
        //     fw.write( "\n" ); 
        //     fw.flush();
        // }
        // fw.write( "   " + best_objectvalue + "   \n" ); fw.flush();

        return particle;
    }
    static void readDataset() throws IOException
	{
		Vector<Vector<Double>> dataset_tmp = new Vector<Vector<Double>>();
        Vector<Double> max_value_v = new Vector<Double>();
        Vector<Double> min_value_v = new Vector<Double>();

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
				if( i > max_value_v.size()-1 )
					max_value_v.add( tmp.get(i) );
				else if( tmp.get(i) > max_value_v.get(i) )
					max_value_v.set(i, tmp.get(i));
				
				if( i > min_value_v.size()-1 )
					min_value_v.add( tmp.get(i) );
				else if( tmp.get(i) < min_value_v.get(i) )
					min_value_v.set(i, tmp.get(i));
			}
		}
		fr.close();
		num_dimension = max_value_v.size();
		num_data = dataset_tmp.size();
        dataset = new Double[num_data][num_dimension];
        max_value = new Double[num_dimension];
        min_value = new Double[num_dimension];
        for( int i=0; i<num_dimension; i++ )
        {
            max_value[i] = max_value_v.get(i);
            min_value[i] = min_value_v.get(i);
        }
        for( int i=0; i<num_data; i++ )
			for( int j=0; j<num_dimension; j++ )
				dataset[i][j] = dataset_tmp.get(i).get(j);
	}
    static double random( double min, double max )
	{
		return Math.random()*(max-min) + min;
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
    static double compute_dist( Double[] data, Double[] center )
	{
		double sum = 0;
		for( int i=0; i<data.length; i++ )
			sum += Math.pow(center[i] - data[i],2);
		return Math.sqrt(sum);
	}
}


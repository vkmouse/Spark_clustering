## Clustering algorithms with Apache Spark
This code implements three clustering algorithms with Apache Spark in Java version, *k*-means, genetic *k*-means algorithm (GKA), and particle swarm optimization (PSO) respectively to solve five datasets of clustering.


 - [Introduction](#Introduction)
 - [Compile](#Compile)
 - [Usage](#Usage)
   - [*k*-means](#k-means)
   - [Genetic *k*-means algorithm](#Genetic k-means algorithm)
   - [Particle swarm optimization](#Particle swarm optimization)
   
### Introduction
[Apache Spark](https://spark.apache.org/) is a distributed general-purpose cluster-computing framework. It also uses MapReduce architecture to realize the distribution program. We expect to save execution time in the larger dataset when we implement the clustering algorithm with Apache Spark. The simulated results [[1]](https://doi.org/10.1016/j.jpdc.2017.10.020) show that GKA with Apache Spark reduce the execution time in the larger dataset.

### Compile
Install Apache Spark 1.5.2 or newer version, Java 1.8.0_144 or newer and Apache Maven. The following example installs Apache Spark with the docker image. So first you have to install docker. 

```bash
apt-get update
apt install docker.io
docker --version
```
    
Second, create a file named **dockerfile** in the root path. Paste the following illustration to the **dockerfile**.

    FROM bde2020/spark-master:2.4.3-hadoop2.7
    RUN apk add --no-cache openjdk8 maven git
    ENV PATH $PATH:/spark/bin
    RUN git clone https://github.com/vkmouse/Spark_clustering.git
    CMD ["/bin/bash"]

Third, build a new image with the dockerfile. This image is built based on [Big Data Europe](https://github.com/big-data-europe/docker-spark).

```bash
docker build --rm=true -t spark-clustering .
```
Fourth, run the image and execute the bash.

```bash
docker run -it --rm=true --name spark-clustering -e ENABLE_INIT_DAEMON=false -d spark-clustering:latest
docker exec -it spark-clustering bash
```
    
Finally, go to the path of **Spark_clustering** and compile it with maven.

```bash
cd Spark_clustering
mvn package
```

### Usage
Five parameters are required for all algorithms to execute the program. Each algorithm requires specific parameters for itself.
1. Dataset path
2. Number of iterations
3. Number of clusters
4. Output filename
5. Number of runs

#### *k*-means
No other parameter is required to execute the *k*-means. For example, dataset path is "dataset/iris.txt", number of iterations is 50, number of clusters is 3, output filename is "kmeans.txt", and number of runs is 2.

```bash
spark-submit --class edu.nchu.app.kmeans target/Spark_clustering-1.0-SNAPSHOT.jar dataset/iris.txt 50 3 kmeans.txt 2
```

#### Genetic *k*-means algorithm
Three parameters are required to execute the GKA. And set the output path to gka.txt.
1. Number of chromsomes
2. Crossover rate
3. Mutation rate

For example, number of chromsomes is 20, crossover rate is 0.6, and mutation rate is 0.9.
 
 ```bash
spark-submit --class edu.nchu.app.gka target/Spark_clustering-1.0-SNAPSHOT.jar dataset/iris.txt 50 3 gka.txt 2 20 0.6 0.9
```


#### Particle swarm optimization
Five parameters are required to execute the PSO. And set the output path to pso.txt.
1. Number of particles
2. c<sub>1</sub>
3. c<sub>2</sub>
4. Maximum of weight
5. Minimum of weight

For example, number of particles is 20, c<sub>1</sub> is 2,  c<sub>2</sub> is 2, maximum of weight is 0.9, and  minimum of weight is 0.4
 
```bash
spark-submit --class edu.nchu.app.pso target/Spark_clustering-1.0-SNAPSHOT.jar dataset/iris.txt 50 3 pso.txt 2 20 2 2 0.9 0.4
```


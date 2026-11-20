import java.io.IOException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * MapReduce Job to calculate average rating and total review counts per movie.
 * Input format: userId,movieId,rating,timestamp
 */
public class MovieRatingStats {

    public static class RatingMapper extends Mapper<LongWritable, Text, Text, Text> {
        private Text movieId = new Text();
        private Text ratingValue = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString().trim();
            if (line.isEmpty() || line.startsWith("userId")) {
                // Skip header or empty lines
                return;
            }

            String[] tokens = line.split(",");
            if (tokens.length >= 3) {
                try {
                    String mId = tokens[1].trim();
                    float rating = Float.parseFloat(tokens[2].trim());

                    movieId.set(mId);
                    ratingValue.set(String.valueOf(rating));
                    context.write(movieId, ratingValue);
                } catch (NumberFormatException e) {
                    // Ignore malformed rows
                }
            }
        }
    }

    public static class RatingReducer extends Reducer<Text, Text, Text, Text> {
        private Text result = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            float sum = 0;
            int count = 0;

            for (Text val : values) {
                sum += Float.parseFloat(val.toString());
                count++;
            }

            if (count > 0) {
                float avg = sum / count;
                result.set(String.format("AvgRating: %.2f\tTotalRatings: %d", avg, count));
                context.write(key, result);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: MovieRatingStats <input path> <output path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Movie Rating Stats");

        job.setJarByClass(MovieRatingStats.class);
        job.setMapperClass(RatingMapper.class);
        job.setReducerClass(RatingReducer.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

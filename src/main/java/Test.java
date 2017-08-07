import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DBOW;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.text.documentiterator.FilenamesLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * TODO: Write class description
 * Created by d814039 on 7/25/2017.
 */
public class Test {

    private static final Logger log = LoggerFactory.getLogger(Test.class);

    private static final List<String> stopWords = Collections.unmodifiableList(Arrays.asList("a", "about", "above",
            "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at", "be", "because",
            "been", "before", "being", "below", "between", "both", "but", "by", "cant", "cannot", "could", "couldnt",
            "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for", "from",
            "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes", "her",
            "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive",
            "if", "in", "into", "is", "isnt", "it", "its", "its", "itself", "lets", "me", "more", "most", "mustnt",
            "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
            "ours", "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should",
            "shouldnt", "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves",
            "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those",
            "through", "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were",
            "weve", "were", "werent", "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who",
            "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre",
            "youve", "your", "yours", "yourself", "yourselves")
    );

    public static void main(String[] args) throws IOException {
        //thing();
        buildFitTest();
        //readFitTest();
        //readTest();
    }

    private static void thing() throws IOException {
        ClassPathResource trainData = new ClassPathResource("train");

        LabelAwareIterator iterator = new FilenamesLabelAwareIterator.Builder()
                .addSourceFolder(trainData.getFile())
                .build();

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                .allowParallelTokenization(true)
                .batchSize(512)
                .elementsLearningAlgorithm(new CBOW<>())
                .epochs(1)
                .iterations(1)
                .iterate(iterator)
                .layerSize(300)
                .learningRate(0.001)
                .minLearningRate(0.0001)
                .minWordFrequency(5)
                .modelUtils(new BasicModelUtils<>())
                .usePreciseWeightInit(false)
                .sampling(0.0)
                .seed(413)
                .sequenceLearningAlgorithm(new DBOW<>())
                .stopWords(stopWords)
                .tokenizerFactory(tokenizerFactory)
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(true)
                .trainWordVectors(true)
                .useAdaGrad(false)
                .useHierarchicSoftmax(true)
                .useUnknown(false)
                .windowSize(5)
                .build();

        paragraphVectors.fit();

        WordVectorSerializer.writeParagraphVectors(paragraphVectors, "vectors.zip");

        // Begin Testing
        ClassPathResource testData = new ClassPathResource("test");

        LabelAwareIterator testIterator = new FilenamesLabelAwareIterator.Builder()
                .addSourceFolder(testData.getFile())
                .build();

        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(), tokenizerFactory);
        LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        while (testIterator.hasNextDocument()) {
            LabelledDocument document = testIterator.nextDocument();
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

            scores.sort(Comparator.comparing(p -> -p.getSecond()));

            log.info("Document '" + document.getLabels() + "' is most similar to:");
            for (int i = 0; i < 10; i++) {
                log.info("    " + scores.get(i).getFirst() + ": " + scores.get(i).getSecond());
            }
        }
    }

    private static void buildFitTest() throws IOException {
        ClassPathResource trainData = new ClassPathResource("train");

        LabelAwareIterator iterator = new FilenamesLabelAwareIterator.Builder()
                .addSourceFolder(trainData.getFile())
                .build();

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                .allowParallelTokenization(true)
                .batchSize(512)
                .elementsLearningAlgorithm(new CBOW<>())
                .epochs(1)
                .iterations(1)
                .iterate(iterator)
                .layerSize(300)
                .learningRate(0.001)
                .minLearningRate(0.0001)
                .minWordFrequency(5)
                .modelUtils(new BasicModelUtils<>())
                .usePreciseWeightInit(false)
                .sampling(0.0)
                .seed(413)
                .sequenceLearningAlgorithm(new DBOW<>())
                .stopWords(stopWords)
                .tokenizerFactory(tokenizerFactory)
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(true)
                .trainWordVectors(true)
                .useAdaGrad(false)
                .useHierarchicSoftmax(true)
                .useUnknown(false)
                .windowSize(5)
                .build();

        paragraphVectors.fit();

        /*
        System.out.println("==================================================================");
        System.out.println(paragraphVectors.nearestLabels("Trump says Nato 'no longer obsolete'\n" +
                "US President Donald Trump has said Nato is \"no longer obsolete\", reversing a stance that had alarmed allies. \n" +
                "Hosting Nato Secretary General Jens Stoltenberg at the White House, Mr Trump said the threat of terrorism had underlined the alliance's importance. \n" +
                "He called on Nato to do more to help Iraqi and Afghan \"partners\". \n" +
                "Mr Trump has repeatedly questioned Nato's purpose, while complaining that the US pays an unfair share of membership. \n" +
                "The Nato U-turn wasn't Mr Trump's only change of heart on Wednesday. \n" +
                "In an interview with the Wall Street Journal, he said he would not label China a currency manipulator, despite having repeatedly pledged to do so on his first day in office. \n" +
                "At a joint press conference with Mr Stoltenberg, Mr Trump said: \"The secretary general and I had a productive discussion about what more Nato can do in the fight against terrorism. \n" +
                "\"I complained about that a long time ago and they made a change, and now they do fight terrorism. \n" +
                "\"I said it [Nato] was obsolete. It's no longer obsolete.\" \n" +
                "But Mr Trump reiterated his call for Nato member states to contribute more funding to the alliance. \n" +
                "\"If other countries pay their fair share instead of relying on the United States to make up the difference we will all be much more secure,\" said the US president. \n" +
                "Mr Stoltenberg thanked Mr Trump for \"an excellent and very productive meeting\". \n" +
                "Earlier this week Nato welcomed Montenegro as its 29th member nation. \n" +
                "The meeting at the White House comes hours after US Secretary of State Rex Tillerson met Russian President Vladimir Putin during a trip to Moscow. \n" +
                "\"Things went pretty well. Maybe better than anticipated,\" Mr Trump said about that meeting. \n" +
                "\"Right now we're not getting along with Russia at all. We may be at an all-time low in terms of relationship with Russia.\" \n", 10));
        System.out.println("==================================================================");
        */

        WordVectorSerializer.writeParagraphVectors(paragraphVectors, "vectors.zip");

        // Begin Testing
        ClassPathResource testData = new ClassPathResource("test");

        LabelAwareIterator testIterator = new FilenamesLabelAwareIterator.Builder()
                .addSourceFolder(testData.getFile())
                .build();

        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(), tokenizerFactory);
        LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        while (testIterator.hasNextDocument()) {
            LabelledDocument document = testIterator.nextDocument();
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

            scores.sort(Comparator.comparing(p -> -p.getSecond()));

            log.info("Document '" + document.getLabels() + "' is most similar to:");
            for (int i = 0; i < 10; i++) {
                log.info("    " + scores.get(i).getFirst() + ": " + scores.get(i).getSecond());
            }
        }
    }

    private static void readTest() throws IOException {
        ClassPathResource trainData = new ClassPathResource("train");

        LabelAwareIterator iterator = new FilenamesLabelAwareIterator.Builder()
                .addSourceFolder(trainData.getFile())
                .build();

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                .allowParallelTokenization(true)
                .batchSize(512)
                .elementsLearningAlgorithm(new SkipGram<>())
                .epochs(5)
                .iterations(1)
                .iterate(iterator)
                .layerSize(10)
                .learningRate(0.001)
                .minLearningRate(0.0001)
                .minWordFrequency(5)
                .modelUtils(new BasicModelUtils<>())
                .usePreciseWeightInit(false)
                .sampling(0.0)
                .seed(413)
                .sequenceLearningAlgorithm(new DBOW<>())
                .stopWords(stopWords)
                .tokenizerFactory(tokenizerFactory)
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(true)
                .trainWordVectors(true)
                .useAdaGrad(false)
                .useHierarchicSoftmax(true)
                .useUnknown(false)
                .windowSize(5)
                .build();

        paragraphVectors = WordVectorSerializer.readParagraphVectors("vectors.zip");

        // Begin Testing
        ClassPathResource testData = new ClassPathResource("test");

        LabelAwareIterator testIterator = new FilenamesLabelAwareIterator.Builder()
                .addSourceFolder(testData.getFile())
                .build();

        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(), tokenizerFactory);
        LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        while (testIterator.hasNextDocument()) {
            LabelledDocument document = testIterator.nextDocument();
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

            scores.sort(Comparator.comparing(p -> -p.getSecond()));

            log.info("Document '" + document.getLabels() + "' is most similar to:");
            for (int i = 0; i < 10; i++) {
                log.info("    " + scores.get(i).getFirst() + ": " + scores.get(i).getSecond());
            }
        }
    }

    public static void readFitTest() throws IOException {
        ClassPathResource trainData = new ClassPathResource("train");

        LabelAwareIterator iterator = new FilenamesLabelAwareIterator.Builder()
                .addSourceFolder(trainData.getFile())
                .build();

        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                .allowParallelTokenization(true)
                .batchSize(512)
                .elementsLearningAlgorithm(new SkipGram<>())
                .epochs(5)
                .iterations(1)
                .iterate(iterator)
                .layerSize(10)
                .learningRate(0.001)
                .minLearningRate(0.0001)
                .minWordFrequency(5)
                .modelUtils(new BasicModelUtils<>())
                .usePreciseWeightInit(false)
                .sampling(0.0)
                .seed(413)
                .sequenceLearningAlgorithm(new DBOW<>())
                .stopWords(stopWords)
                .tokenizerFactory(tokenizerFactory)
                .trainElementsRepresentation(true)
                .trainSequencesRepresentation(true)
                .trainWordVectors(true)
                .useAdaGrad(false)
                .useHierarchicSoftmax(true)
                .useUnknown(false)
                .windowSize(5)
                .build();

        paragraphVectors = WordVectorSerializer.readParagraphVectors("vectors.zip");

        paragraphVectors.fit();

        // Begin Testing
        ClassPathResource testData = new ClassPathResource("test");

        LabelAwareIterator testIterator = new FilenamesLabelAwareIterator.Builder()
                .addSourceFolder(testData.getFile())
                .build();

        MeansBuilder meansBuilder = new MeansBuilder(
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable(), tokenizerFactory);
        LabelSeeker seeker = new LabelSeeker(iterator.getLabelsSource().getLabels(),
                (InMemoryLookupTable<VocabWord>) paragraphVectors.getLookupTable());

        while (testIterator.hasNextDocument()) {
            LabelledDocument document = testIterator.nextDocument();
            INDArray documentAsCentroid = meansBuilder.documentAsVector(document);
            List<Pair<String, Double>> scores = seeker.getScores(documentAsCentroid);

            scores.sort(Comparator.comparing(p -> -p.getSecond()));

            log.info("Document '" + document.getLabels() + "' is most similar to:");
            for (int i = 0; i < 10; i++) {
                log.info("    " + scores.get(i).getFirst() + ": " + scores.get(i).getSecond());
            }
        }
    }

}

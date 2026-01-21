import Foundation
import CreateML
import CoreML

/// Trains recommendation models using collaborative filtering
public class RecommenderTrainer {

    public struct TrainingParameters {
        public var userColumn: String
        public var itemColumn: String
        public var ratingColumn: String?
        public var validationData: URL?

        public init(
            userColumn: String = "user",
            itemColumn: String = "item",
            ratingColumn: String? = nil,
            validationData: URL? = nil
        ) {
            self.userColumn = userColumn
            self.itemColumn = itemColumn
            self.ratingColumn = ratingColumn
            self.validationData = validationData
        }
    }

    public struct TrainingResult {
        public let modelURL: URL
        public let trainingRMSE: Double?
        public let validationRMSE: Double?
        public let trainingDuration: TimeInterval
    }

    public init() {}

    /// Train a recommender from interaction data
    /// - Parameters:
    ///   - trainingDataURL: CSV or JSON file with user-item interactions
    ///   - outputURL: Where to save the trained .mlmodel
    ///   - parameters: Training parameters
    ///   - progressHandler: Called with progress updates
    /// - Returns: Training result with metrics
    public func train(
        trainingDataURL: URL,
        outputURL: URL,
        author: String? = nil,
        description: String? = nil,
        parameters: TrainingParameters,
        progressHandler: ((String) -> Void)? = nil
    ) throws -> TrainingResult {
        let startTime = Date()

        progressHandler?("Loading training data from \(trainingDataURL.path)...")

        let trainingData = try MLDataTable(contentsOf: trainingDataURL)

        progressHandler?("Found \(trainingData.rows.count) interactions...")

        progressHandler?("Training recommender model...")

        let recommender: MLRecommender

        if let ratingCol = parameters.ratingColumn {
            // Explicit ratings (e.g., 1-5 stars)
            recommender = try MLRecommender(
                trainingData: trainingData,
                userColumn: parameters.userColumn,
                itemColumn: parameters.itemColumn,
                ratingColumn: ratingCol
            )
        } else {
            // Implicit feedback (interactions without ratings)
            recommender = try MLRecommender(
                trainingData: trainingData,
                userColumn: parameters.userColumn,
                itemColumn: parameters.itemColumn
            )
        }

        progressHandler?("Saving model to \(outputURL.path)...")

        let metadata = MLModelMetadata(
            author: author ?? "CreateML CLI",
            shortDescription: description ?? "Recommender trained with CreateML CLI",
            version: "1.0"
        )

        try recommender.write(to: outputURL, metadata: metadata)

        let trainingDuration = Date().timeIntervalSince(startTime)

        return TrainingResult(
            modelURL: outputURL,
            trainingRMSE: nil,
            validationRMSE: nil,
            trainingDuration: trainingDuration
        )
    }
}

//
//  BGEEmbeddingModel.swift
//  SwiftRAGEngine
//
//  Created by Andrew Wallace on 26/08/26.
//

import CoreML

final class BGEEmbeddingModel {
    private let model: BGE

    init() throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all

        model = try BGE(
            configuration: configuration
        )
    }
    
    func embed(
        inputIDs: [Int32],
        attentionMask: [Int32]
    ) throws -> [Float] {
        let inputIDsArray = try MLMultiArray(
            shape: [1, 128],
            dataType: .int32
        )
        
        let attentionMaskArray = try MLMultiArray(
            shape: [1, 128],
            dataType: .int32
        )
        
        for index in 0..<128 {
            inputIDsArray[index] = NSNumber(
                value: inputIDs[index]
            )
            
            attentionMaskArray[index] = NSNumber(
                value: attentionMask[index]
            )
        }
        
        // Run Core ML
        let output = try model.prediction(
            input_ids: inputIDsArray, attention_mask: attentionMaskArray
        )
        
        // Convert MLMultiArray -> [Float]
        let embedding = (0..<output.embedding.count).map {
            output.embedding[$0].floatValue
        }
        
        return embedding
    }
}

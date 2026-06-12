package com.example.demo.repository;

import com.example.demo.model.RecommendationUnexpected;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface RecommendationUnexpectedRepository extends MongoRepository<RecommendationUnexpected, String> {
}

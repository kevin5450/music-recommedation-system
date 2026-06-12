package com.example.demo.repository;

import com.example.demo.model.RecommendationPersonalPage;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface RecommendationPersonalPageRepository extends MongoRepository<RecommendationPersonalPage, String> {
}

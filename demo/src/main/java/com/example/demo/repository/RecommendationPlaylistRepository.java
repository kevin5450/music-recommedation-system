package com.example.demo.repository;

import com.example.demo.model.RecommendationPlaylist;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface RecommendationPlaylistRepository extends MongoRepository<RecommendationPlaylist, String> {
}

package com.example.demo.repository;

import com.example.demo.model.FirstFiveSongs;
import org.springframework.data.mongodb.repository.MongoRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface FirstFiveSongsRepository extends MongoRepository<FirstFiveSongs, String> {
    List<FirstFiveSongs> findTop5ByOrderByIdAsc();
}

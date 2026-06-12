package com.example.demo.model;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Document(collection = "music") // MongoDB 컬렉션 이름 지정
public class Music {
    @Id
    private String id;
    private String title;
    private String artist;
    private int popularity;

    public Music() {}

    public Music(String title, String artist, int popularity) {
        this.title = title;
        this.artist = artist;
        this.popularity = popularity;
    }


    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getTitle() {
        return title;
    }

    public void setTitle(String title) {
        this.title = title;
    }

    public String getArtist() {
        return artist;
    }

    public void setArtist(String artist) {
        this.artist = artist;
    }

    public int getPopularity() { return popularity; }
    public void setPopularity(int popularity) { this.popularity = popularity; }
}

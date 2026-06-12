package com.example.demo.controller;

import com.example.demo.model.User;
import com.example.demo.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import java.util.Optional;

@Controller
public class LoginController {

    @Autowired
    private UserRepository userRepository;

    @PostMapping("/login")
    public String login(
            @RequestParam("userid") String userid,
            @RequestParam("password") String password,
            Model model) {

        Optional<User> userOptional = userRepository.findByUserid(userid);

        // 아이디 존재 확인
        if (userOptional.isEmpty()) {
            model.addAttribute("error", "아이디가 존재하지 않습니다.");
            model.addAttribute("userid", userid); // 입력값 유지
            return "login";
        }

        User user = userOptional.get();

        // 비밀번호 확인
        if (!user.getPassword().equals(password)) {
            model.addAttribute("error", "비밀번호가 올바르지 않습니다.");
            model.addAttribute("userid", userid); // 입력값 유지
            return "login";
        }

        // 로그인 성공 (메인 페이지로 이동)
        return "redirect:/Mainpage";
    }
}

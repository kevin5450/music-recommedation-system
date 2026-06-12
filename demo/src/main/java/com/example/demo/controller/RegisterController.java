package com.example.demo.controller;

import com.example.demo.model.User;
import com.example.demo.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;

@Controller
public class RegisterController {

    @Autowired
    private UserRepository userRepository;

    // 회원가입 페이지 요청 (GET)
    @GetMapping("/register")
    public String showRegisterPage() {
        return "register";
    }

    // 회원가입 처리 (POST)
    @PostMapping("/register")
    public String registerUser(
            @RequestParam("userid") String userid,
            @RequestParam("fullname") String fullname,
            @RequestParam("email") String email,
            @RequestParam("password") String password,
            @RequestParam("passwordConfirm") String passwordConfirm,
            @RequestParam("nickname") String nickname,
            @RequestParam("birthdate") String birthdate,
            @RequestParam("phone") String phone,
            @RequestParam(value = "newsletter", defaultValue = "false") boolean newsletter,
            Model model) {


        model.addAttribute("userid", userid);
        model.addAttribute("fullname", fullname);
        model.addAttribute("email", email);
        model.addAttribute("password", password);
        model.addAttribute("passwordConfirm", passwordConfirm);
        model.addAttribute("nickname", nickname);
        model.addAttribute("birthdate", birthdate);
        model.addAttribute("phone", phone);
        model.addAttribute("newsletter", newsletter);

        if (userRepository.existsByUserid(userid)) {
            model.addAttribute("error", "이미 존재하는 아이디입니다.");
            return "register";
        }
        if (userRepository.existsByEmail(email)) {
            model.addAttribute("error", "이미 사용 중인 이메일입니다.");
            return "register";
        }

        // 비밀번호 확인
        if (!password.equals(passwordConfirm)) {
            model.addAttribute("error", "비밀번호가 일치하지 않습니다.");
            return "register";
        }

        // User 객체 생성 및 저장
        User newUser = new User(userid, fullname, email, password, nickname, birthdate, phone, newsletter);
        userRepository.save(newUser);

        return "redirect:/profile-completion";
    }
}

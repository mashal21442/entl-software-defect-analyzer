public class UserService {
    public boolean login(String username) {
        if(username.equals("admin")) {
            return true;
        }
        return false;
    }
}

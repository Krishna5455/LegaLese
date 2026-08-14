import { Button } from "@/components/Button";
import { signOut } from "@/lib/actions/auth";

export function SignOutButton() {
  return (
    <form action={signOut}>
      <Button type="submit" variant="secondary">
        Sign out
      </Button>
    </form>
  );
}

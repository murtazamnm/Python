
# You are given two non-empty linked lists representing two non-negative integers. 
# The digits are stored in reverse order, and each of their nodes contains a single digit. 
# Add the two numbers and return the sum as a linked list.
# You may assume the two numbers do not contain any leading zero, except the number 0 itself.

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        st = ""
        current_node = l1
        while current_node is not None:
            st+=str(current_node.val)
            current_node = current_node.next
        st+=" "
        current_node = l2
        while current_node is not None:
            st+=str(current_node.val)
            current_node = current_node.next
        res = int(st.split(" ")[0][::-1]) + int(st.split(" ")[1][::-1])
        res = str(res)[::-1]
        ls = [int(x) for x in str(res)]
        lnls = ListNode(ls[0])
        c = lnls
        for i in range(1,len(ls)):
            newNode = ListNode(ls[i])
            c.next = newNode
            c = newNode
        return lnls
        
            
        

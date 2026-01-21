// Copyright © Anthony DePasquale

import Foundation

func generateShortId() -> String {
  let characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
  return String((0 ..< 6).map { _ in characters.randomElement()! })
}

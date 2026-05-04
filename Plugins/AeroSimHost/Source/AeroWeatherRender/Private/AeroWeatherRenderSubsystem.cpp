#include "AeroWeatherRenderSubsystem.h"

#include "Dom/JsonObject.h"
#include "Misc/FileHelper.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Weather/WeatherLib.h"

namespace
{
bool LoadJsonObjectFromFile(const FString& FilePath, TSharedPtr<FJsonObject>& OutObject, FString& OutError)
{
	FString Content;
	if (!FFileHelper::LoadFileToString(Content, *FilePath))
	{
		OutError = FString::Printf(TEXT("Failed to read JSON file: %s"), *FilePath);
		return false;
	}

	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Content);
	if (!FJsonSerializer::Deserialize(Reader, OutObject) || !OutObject.IsValid())
	{
		OutError = FString::Printf(TEXT("Failed to parse JSON file: %s"), *FilePath);
		return false;
	}

	return true;
}

float ReadWeatherScalar(const TSharedPtr<FJsonObject>& Snapshot, const TSharedPtr<FJsonObject>& Profile, const FString& FieldName, float DefaultValue = 0.0f)
{
	if (Snapshot.IsValid() && Snapshot->HasField(FieldName))
	{
		return static_cast<float>(Snapshot->GetNumberField(FieldName));
	}
	if (Profile.IsValid() && Profile->HasField(FieldName))
	{
		return static_cast<float>(Profile->GetNumberField(FieldName));
	}
	return DefaultValue;
}
}

bool UAeroWeatherRenderSubsystem::ShouldCreateSubsystem(UObject* Outer) const
{
	const UWorld* World = Cast<UWorld>(Outer);
	return World != nullptr && World->IsGameWorld();
}

bool UAeroWeatherRenderSubsystem::LoadProfiles(const FString& ProfilesPath, FString& OutError)
{
	return LoadJsonObjectFromFile(ProfilesPath, ProfilesDocument, OutError);
}

TSharedPtr<FJsonObject> UAeroWeatherRenderSubsystem::ApplyWeather(const TSharedPtr<FJsonObject>& Payload, FString& OutError)
{
	if (!Payload.IsValid())
	{
		OutError = TEXT("ApplyWeather payload is invalid.");
		return nullptr;
	}

	UWorld* World = GetWorld();
	if (World == nullptr)
	{
		OutError = TEXT("No valid UWorld for weather rendering.");
		return nullptr;
	}

	FString Condition = TEXT("clear");
	Payload->TryGetStringField(TEXT("condition"), Condition);

	TSharedPtr<FJsonObject> ProfileObject;
	if (ProfilesDocument.IsValid() && ProfilesDocument->HasTypedField<EJson::Object>(TEXT("profiles")))
	{
		const TSharedPtr<FJsonObject> ProfilesMap = ProfilesDocument->GetObjectField(TEXT("profiles"));
		if (ProfilesMap.IsValid())
		{
			if (ProfilesMap->HasTypedField<EJson::Object>(Condition))
			{
				ProfileObject = ProfilesMap->GetObjectField(Condition);
			}
			else if (ProfilesMap->HasTypedField<EJson::Object>(TEXT("default")))
			{
				ProfileObject = ProfilesMap->GetObjectField(TEXT("default"));
			}
		}
	}

	UWeatherLib::setWeatherEnabled(World, true);
	UWeatherLib::setWeatherParamScalar(World, EWeatherParamScalar::WEATHER_PARAM_SCALAR_RAIN, ReadWeatherScalar(Payload, ProfileObject, TEXT("rain")));
	UWeatherLib::setWeatherParamScalar(World, EWeatherParamScalar::WEATHER_PARAM_SCALAR_ROADWETNESS, ReadWeatherScalar(Payload, ProfileObject, TEXT("wetness")));
	UWeatherLib::setWeatherParamScalar(World, EWeatherParamScalar::WEATHER_PARAM_SCALAR_FOG, ReadWeatherScalar(Payload, ProfileObject, TEXT("fog_density")));
	UWeatherLib::setWeatherParamScalar(World, EWeatherParamScalar::WEATHER_PARAM_SCALAR_DUST, ReadWeatherScalar(Payload, ProfileObject, TEXT("dust")));

	if (Payload->HasTypedField<EJson::Array>(TEXT("wind_vector_enu_mps")))
	{
		const TArray<TSharedPtr<FJsonValue>>& WindValues = Payload->GetArrayField(TEXT("wind_vector_enu_mps"));
		if (WindValues.Num() >= 3)
		{
			UWeatherLib::setWeatherWindDirection(World, FVector(WindValues[0]->AsNumber(), WindValues[1]->AsNumber(), WindValues[2]->AsNumber()));
		}
	}

	TSharedPtr<FJsonObject> Result = MakeShared<FJsonObject>();
	Result->SetStringField(TEXT("condition"), Condition);
	Result->SetBoolField(TEXT("applied"), true);
	Result->SetNumberField(TEXT("rain"), ReadWeatherScalar(Payload, ProfileObject, TEXT("rain")));
	Result->SetNumberField(TEXT("wetness"), ReadWeatherScalar(Payload, ProfileObject, TEXT("wetness")));
	Result->SetNumberField(TEXT("fog_density"), ReadWeatherScalar(Payload, ProfileObject, TEXT("fog_density")));
	return Result;
}
